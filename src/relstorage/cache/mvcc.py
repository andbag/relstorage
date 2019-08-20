# -*- coding: utf-8 -*-
##############################################################################
#
# Copyright (c) 2019 Zope Foundation and Contributors.
# All Rights Reserved.
#
# This software is subject to the provisions of the Zope Public License,
# Version 2.1 (ZPL).  A copy of the ZPL should accompany this distribution.
# THIS SOFTWARE IS PROVIDED "AS IS" AND ANY AND ALL EXPRESS OR IMPLIED
# WARRANTIES ARE DISCLAIMED, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF TITLE, MERCHANTABILITY, AGAINST INFRINGEMENT, AND FITNESS
# FOR A PARTICULAR PURPOSE.
#
##############################################################################
"""
MVCC tracking for cache objects.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import threading
import weakref


from zope.interface import implementer

from relstorage._compat import iterkeys
from relstorage._compat import OID_OBJECT_MAP_TYPE as OidOMap
from relstorage._compat import OID_TID_MAP_TYPE as OidTMap
from relstorage._compat import OID_SET_TYPE as OidSet
from relstorage._compat import iteroiditems
from relstorage._util import log_timed

from .interfaces import IStorageCacheMVCCDatabaseCoordinator
from ._util import InvalidationMixin

logger = __import__('logging').getLogger(__name__)

###
# Notes on in-process concurrency:
#
# Where possible, we rely on atomic primitive operations of fundamental types.
# For example, we rely on ``dict[key] = value`` being atomic.
# On CPython we use BTrees which are implemented in C and don't release the GIL
# because they're always resident, so this holds true. On PyPy we use
# native dicts and this also holds true. On CPython if we used PURE_PYTHON
# BTrees, this would *not* hold true, so we also use dicts.
###

class _TransactionRangeObjectIndex(OidTMap):
    """
    Holds the portion of the object index visible to transactions <=
    ``highest_visible_tid``.

    Initialized to be either empty, or with the *complete* index of
    objects <= ``highest_visible_tid`` and greater than
    ``complete_since_tid``. (That is, you cannot pass an 'ignore_txn'
    value to the poller. TODO: Workon that. Maybe there's a way.)

    ``highest_visible_tid`` must always be given, but
    ``complete_since_tid`` may be null.

    These attribute, once set, will never change. We may add data
    prior to and including ``complete_since_tid`` as we access it, but we have no
    guarantee that it is complete.

    When ``complete_since_tid`` and ``highest_visible_tid`` are the same
    """

    __slots__ = (
        'highest_visible_tid',
        'complete_since_tid',
    )

    def __init__(self, highest_visible_tid, complete_since_tid, data):
        assert complete_since_tid is None or highest_visible_tid >= complete_since_tid
        self.highest_visible_tid = highest_visible_tid
        self.complete_since_tid = complete_since_tid

        OidTMap.__init__(self, data)

        if self:
            # Verify the data matches what they told us.
            # If we were constructed with data, we must be complete.
            # Otherwise we get built up bit by bit.
            assert self.complete_since_tid
            self.verify()

    def verify(self, initial=True):
        # Check that our constraints are met
        if not self or not __debug__:
            return

        mxv = self.maxValue()
        mnv = self.minValue()
        assert mxv <= self.highest_visible_tid, (mxv, self.highest_visible_tid)
        assert mnv > 0, mnv
        if initial:
            # This is only true at startup. Over time we can add older entries.
            assert self.complete_since_tid is None or mnv > self.complete_since_tid, (
                mnv, self.complete_since_tid)

    def complete_to(self, newer_bucket):
        """
        Given an incomplete bucket (this object) and a complete bucket for the
        same or a later TID, merge this one to hold the same data and be complete
        for the same transaction range.
        """
        assert not self.complete_since_tid
        assert newer_bucket.complete_since_tid
        assert newer_bucket.highest_visible_tid >= self.highest_visible_tid
        self.update(newer_bucket)
        self.highest_visible_tid = newer_bucket.highest_visible_tid
        self.complete_since_tid = newer_bucket.complete_since_tid

    def merge_same_tid(self, bucket):
        """
        Given an incoming complete bucket for the same highest tid as this bucket,
        merge the two into this object.
        """
        assert bucket.highest_visible_tid == self.highest_visible_tid
        self.update(bucket)
        if bucket.complete_since_tid < self.complete_since_tid:
            self.complete_since_tid = bucket.complete_since_tid

    def merge_older_tid(self, bucket):
        """
        Given an incoming complete bucket for an older tid than this bucket,
        merge the two into this object.

        Because we're newer, objects in this bucket supercede objects
        in the incoming data.
        """
        assert bucket.highest_visible_tid <= self.highest_visible_tid
        bucket.update(self) # overwrite with our data
        self.update(bucket) # bring back to ourself.
        if bucket.complete_since_tid < self.complete_since_tid:
            self.complete_since_tid = bucket.complete_since_tid

    def close(self):
        self.highest_visible_tid = -1
        self.clear()

    # These raise ValueError if the map is empty
    if not hasattr(OidTMap, 'maxKey'):
        maxKey = lambda self: max(iterkeys(self))

    if hasattr(OidTMap, 'itervalues'): # BTrees, or Python 2 dict
        def maxValue(self):
            return max(self.itervalues())
        def minValue(self):
            return min(self.itervalues())
    else:
        def maxValue(self):
            return max(self.values())
        def minValue(self):
            return min(self.values())

    def __repr__(self):
        return '<%s at 0x%x hvt=%s complete_after=%s len=%s>' % (
            self.__class__.__name__,
            id(self),
            self.highest_visible_tid,
            self.complete_since_tid,
            len(self),
        )


class _ObjectIndex(object):
    """
    Collectively holds all the object index visible to transactions <=
    ``highest_visible_tid``.

    The entries in ``maps`` attribute are instances of
    :class:`_TransactionRangeObjectIndex`, and our
    ``highest_visible_tid`` is exactly that of the frontmost map. For
    convenience, we also define this to be our
    ``maximum_highest_visible_tid``. The ``highest_visible_tid`` of
    the final map defines our ``minimum_highest_visible_tid``.

    The following describes the general evolution of this index and
    its MVCC strategy. Some of this is implemented here, but much of
    it is implemented in the :class:`MVCCDatabaseCoordinator`.

    .. rubric:: Initial State

    In the beginning, the frontmost map will be empty and incomplete;
    it will have a ``highest_visible_tid`` but no
    ``complete_since_tid``. (Alternately, if we restored from cache,
    we may have data, but we are definitely still incomplete.) We may
    load new object indices into this map, but only if they're less
    than what our ``highest_visible_tid`` is (this is just a special
    version of the general rule for updates laid out below).

    When a new poll takes place and produces a new change list, in the
    form of a :class:`_TransactionRangeObjectIndex`, it is merged with
    the previous map and replaced into this object. This map does have
    a ``complete_since_tid.``

    After this first map is added, moving forward there should be no
    gaps in ``complete_since_tid`` coverage. There may be overlaps,
    but there must be no gaps. Thus, we have complete coverage from
    ``highest_visible_tid`` in the first map all the way back to
    ``complete_since_tid`` in the second-to-last map.

    If the front of this instance already has data matching that same
    ``highest_visible_tid``, the two maps are merged together
    preserving the lowest ``complete_since_tid`` value. In this case,
    a new instance is *not* returned; the same instance is returned so
    that this map can be shared.

    If the front of this instance has a ``highest_visible_tid`` *less*
    than the new polled tid (that is, there's a new TID in the
    database) all data in the new map with a TID less than the current
    ``highest_visible_tid`` could be discarded (but since we walk the
    maps from front to back to find entries, the more data up front
    the better). In fact, we could limit the poll to just that range
    (use ``highest_visible_tid`` as our ``prev_polled_tid``
    parameter), if we're willing to walk through the current map and
    find all objects whose TID is greater than what we would have used
    for ``prev_polled_tid`` (assuming it's >= the current
    ``complete_since_tid`` value) --- but it's probably better to just
    let the database do that.

    On the chance that the value we wish to poll from
    (``prev_polled_tid``) is <= our current ``complete_since_tid``,
    the new map can completely replace the current front of the map
    (merging in any extra data).

    Instead of updating the frontmost map when new a new entry is
    presented, we update the oldest map that can see the new entry.
    The older (farther to the right, or towards the end of the
    ``maps`` list), the more likely it is to be shared by the most
    transactions. (That does have the unfortunate effect of making it
    take longer to get a hit since we have to probe further back in
    the map; maybe we want to update both frontmost and backmost?)

    When a transaction ends and polls, such that the
    ``MVCCDatabaseCoordinator.minimum_highest_visible_tid`` increments
    (which also corresponds to the last reference to that
    ``TransactionRangeObjectIndex`` going away), that map is removed
    from the list (where it was the end), and any unique data in it is
    added to the map at the new back of the list. (The object that was
    removed is *cleared* and marked as *inactive*: other transactions
    will still have that same object on their list, but they shouldn't
    use it anymore to store new index entries.)

    This last map will keep growing. We shrink it in two ways:

        - First, when an OID/TID pair is evicted from the local cache,
          we remove that OID from all maps.

        - Second, when we merge the back two maps, any OID whose TID
          is less than the new ``minimum_highest_visible_tid`` and
          which has no entry in any other map is removed. Crucially,
          it is re-cached with a special key indicating that it hasn't
          changed in a long time; operations that update the cache
          with new data must invalidate this special key. If an object
          is not found in this index, it can then be looked for under
          this special key. Even though the index data is not
          complete, this "frozen" object can still be found.

    When merging this last map, we have an excellent time to purge
    entries from the cache: Any OIDs stored in the last map who have
    entries in other maps has their stored OID/TID from the last map
    removed. These have changed, and by definition their old state is
    not visible to any other connection.

    TODO: Maybe we actually want to use the new ``complete_since_tid``
    value?

    TODO: The merging and recaching might be expensive? Perhaps we
    want to defer it until some threshold is reached?
    """

    __slots__ = (
        'maps',
    )

    def __init__(self, highest_visible_tid, complete_since_tid=None, data=()):
        """
        An instance is created with the first poll, giving us our
        initial TID. It may optionally have data retrieved from
        previously saving the map.
        """
        initial_bucket = _TransactionRangeObjectIndex(highest_visible_tid, complete_since_tid, ())
        initial_bucket.update(data)
        initial_bucket.verify(initial=False)
        # Maps are read from 0...N, so newest bucket must be first.
        self.maps = [initial_bucket]

    def __repr__(self):
        return '<%s at 0x%x maxhvt=%s minhvt=%s depth=%s>' % (
            self.__class__.__name__,
            id(self),
            self.maximum_highest_visible_tid,
            self.minimum_highest_visible_tid,
            len(self.maps),
        )

    def __getitem__(self, oid):
        for mapping in self.maps:
            try:
                # TODO: Microbenchmark this. Which is faster,
                # try/catch or .get()?
                return mapping[oid]
            except KeyError:
                pass
        # No data. Could it be frozen? We'll let the caller decide.
        return None

    def __contains__(self, oid):
        return any(oid in m for m in self.maps)

    def __setitem__(self, oid, tid):
        # Silently discard things that are too new.
        # Something like loadSerial for conflict resolution might do this;
        # we'll see the object from the future when we poll for changes.

        for mapping in reversed(self.maps):
            if tid <= mapping.highest_visible_tid:
                assert mapping.complete_since_tid is None or tid <= mapping.complete_since_tid
                mapping[oid] = tid
                # print("Setting item", oid, tid, mapping)
                break
            # TODO: Do we want to store it everywhere possible for speed,
            # at the cost of memory?

    @property
    def highest_visible_tid(self):
        return self.maps[0].highest_visible_tid

    maximum_highest_visible_tid = highest_visible_tid

    @property
    def minimum_highest_visible_tid(self):
        return self.maps[-1].highest_visible_tid

    @property
    def complete_since_tid(self):
        return self.maps[-1].complete_since_tid

    def invalidate(self, oid_int, tid_int):
        bound = tid_int + 1
        for mapping in self.maps:
            if mapping.get(oid_int, bound) <= tid_int:
                del mapping[oid_int]

    def invalidate_all(self, oids):
        for mapping in self.maps:
            for oid in oids:
                mapping.pop(oid, None)

    def verify(self):
        # Each component has values in range.
        for ix in self.maps:
            ix.verify(initial=False)
        # No gaps in completion
        map_iter = iter(self.maps)
        newest_map = next(map_iter)
        for mapping in map_iter:
            assert newest_map.complete_since_tid <= mapping.highest_visible_tid
            newest_map = mapping

    def with_polled_changes(self, highest_visible_tid, complete_since_tid, changes):
        # Never call this when the poller has specifically said
        # that there are no changes; either the very first poll, or
        # we went backwards due to reverting to a stale state. That's
        # handled at a higher layer.
        assert changes is not None
        assert highest_visible_tid >= self.highest_visible_tid
        assert complete_since_tid is not None

        # First, create the transaction map.
        assert highest_visible_tid and complete_since_tid
        incoming_bucket = _TransactionRangeObjectIndex(highest_visible_tid,
                                                       complete_since_tid,
                                                       changes)
        newest_bucket = self.maps[0]
        oldest_bucket = self.maps[-1]

        # Was this our first poll?
        if newest_bucket is oldest_bucket and oldest_bucket.complete_since_tid is None:
            # Merge the two together and replace ourself.
            assert highest_visible_tid >= oldest_bucket.highest_visible_tid
            # Overwrite the incomplete data with the new data, which is complete
            # back to a point. The make the incomplete data become the complete data
            # print("Oldest bucket", oldest_bucket, "completing to", incoming_bucket)
            oldest_bucket.complete_to(incoming_bucket)
            return self

        # Special cases:
        #
        # - len(new_bucket) == 0: No changes. Therefore, if this was
        #   not our first poll, it must have returned the exact same
        #   highest_visible_tid as our current highest visible tid. We
        #   just need to set our frontmost map's
        #   ``complete_since_tid`` to the lowest of the two values.
        #   This is just a degenerate case of polling to a matching highest_visible_tid.

        # We can't get a higher transaction id unless something changed.
        # put the other way, if nothing changed, it must be for our existing tid.
        __traceback_info__ = incoming_bucket, newest_bucket
        assert incoming_bucket or (
            not incoming_bucket and highest_visible_tid == newest_bucket.highest_visible_tid
        )

        if highest_visible_tid == newest_bucket.highest_visible_tid:
            # print("Merging for same tid", newest_bucket, "incoming", incoming_bucket)
            newest_bucket.merge_same_tid(incoming_bucket)
            return self

        # all that's left is to put the new bucket on front of a new object.
        other = _ObjectIndex.__new__(_ObjectIndex)
        other.maps = [incoming_bucket]
        other.maps.extend(self.maps)
        other.verify() # XXX: Remove before release.
        return other


@implementer(IStorageCacheMVCCDatabaseCoordinator)
class MVCCDatabaseCoordinator(InvalidationMixin):
    """
    Keeps track of the most recent polling data so that
    instances don't make unnecessary polls.

    Shared between all instances of a StorageCache
    in a tree, from the master down.
    """

    object_index = None

    def __init__(self):
        # Use this lock when we're doing normal poll updates
        # or need to read consistent metadata.
        self._da0_lock = threading.Lock()
        self.registered_viewers = []

    def register(self, cache):
        self.registered_viewers.append(weakref.ref(cache, self.registered_viewers.remove))

    def unregister(self, cache):
        if self.is_registered(cache):
            self.registered_viewers.remove(weakref.ref(cache))

    def is_registered(self, cache):
        return weakref.ref(cache) in self.registered_viewers

    @property
    def maximum_highest_visible_tid(self):
        # Visible to *any* connection.
        return None if self.object_index is None else self.object_index.maximum_highest_visible_tid

    @property
    def minimum_highest_visible_tid(self):
        return None if self.object_index is None else self.object_index.minimum_highest_visible_tid

    @property
    def complete_since_tid(self):
        return -1 if self.object_index is None else self.object_index.complete_since_tid

    def debug_info(self):
        return {
            'object_index': self.object_index,
        }

    def poll(self, cache, conn, cursor):
        # For now, we'll do it all while locked, ensuring a linear history
        with self._da0_lock:
            return self._poll_locked(cache, conn, cursor)

    def _poll_locked(self, cache, conn, cursor):
        # Note that poll_invalidations can raise StaleConnectionError,
        # or it can return (None, old_tid) where old_tid is less than
        # its third parameter (``prev_polled_tid``)
        # print("Entering poll")
        if self.object_index is None:
            # print("Initial poll for", cache)
            # Initial poll for the world. Do this locked to be sure we
            # have a consistent state.
            change_iter, tid = cache.adapter.poller.poll_invalidations(conn, cursor, None, None)
            assert change_iter is None
            if tid > 0:
                # print("Found data; can begin caching", cache, tid)
                # tid 0 is empty database, no data.
                self.object_index = _ObjectIndex(tid)
            # print("Initial got tid", tid, self.object_index)
            return change_iter, tid or None, self.object_index or {}


        # We've been here before. See comments in _ObjectIndex docstring about
        # possible optimizations.
        polling_since = cache.highest_visible_tid or self.maximum_highest_visible_tid
        change_iter, tid = cache.adapter.poller.poll_invalidations(
            conn, cursor,
            polling_since,
            None)
        if tid == 0 or tid < polling_since:
            assert change_iter is None
            # print("Freshly zapped or empty db", tid, polling_since, change_iter)
            # Freshly zapped or empty database (tid==0)
            # or stale and asked to revert
            self.object_index = None
            return change_iter, tid, {}

        # Ok cool, we got data to move us forward.
        # Let's do it.
        # Will need to be able to iterate this more than once.
        change_iter = list(change_iter)
        # print("Changes since", polling_since, "now at", tid, "are", change_iter)
        prev_index = self.object_index
        ix_up = _ObjectIndex if prev_index is None else prev_index.with_polled_changes
        self.object_index = new_index = ix_up(tid, polling_since, change_iter)

        if len(new_index.maps) < 2:
            # print("Nothing to freeze", new_index)
            return change_iter, tid, new_index

        prev_min_highest_visible_tid = prev_index.minimum_highest_visible_tid
        # Should we move the minimum forward? Lets see, did we just update the minimum?
        move_forward = prev_min_highest_visible_tid == polling_since
        if move_forward:
            # We did. Were we the one and only one at that minimum?

            for wref in self.registered_viewers:
                viewer = wref()
                if viewer is None or viewer is cache or not viewer.highest_visible_tid:
                    continue
                assert viewer.highest_visible_tid >= polling_since, (viewer, polling_since)
                if viewer == polling_since:
                    # Snarf. Someone else still looking at this.
                    move_forward = False
                    break
        if move_forward:
            # print("Moving forward from", prev_min_highest_visible_tid, "to", tid)
            oldest_bucket = new_index.maps[-2]
            obsolete_bucket = new_index.maps.pop()
            # print("Merging together", oldest_bucket, "with obsolete", obsolete_bucket)
            # print("Oldest data", dict(oldest_bucket))
            # print("Obsolete data", dict(obsolete_bucket))
            oldest_bucket.merge_older_tid(obsolete_bucket)
            # print("Did merge", oldest_bucket, dict(oldest_bucket))
            # XXX: TODO: Here is where we should freeze things.
            # Remove old TIDs from the map, and re-cache objects with the new key.
            obsolete_bucket.close()
            # print("Closed", obsolete_bucket, dict(obsolete_bucket))

        # print("After freeze; changes", change_iter, "new tid", tid, "index", new_index)
        return change_iter, tid, new_index

    def flush_all(self):
        self.object_index = None

    def close(self):
        self._da0_lock = None
        self.object_index = None

    def invalidate_all(self, oids):
        if not self.object_index:
            return

        self.object_index.invalidate_all(oids)

    def invalidate(self, oid_int, tid_int):
        if not self.object_index:
            return
        self.object_index.invalidate(oid_int, tid_int)

    def restore(self, adapter, local_client):
        # This method is not thread safe

        # Note that there may have been a tiny amount of data in the
        # file that we didn't get to actually store but that still
        # comes back in the delta_map; that's ok.
        row_filter = _PersistentRowFilter(adapter,
                                          lambda: _TransactionRangeObjectIndex(1, None, ()))
        local_client.restore(row_filter)
        local_client.remove_invalid_persistent_oids(row_filter.polled_invalid_oids)

        checkpoints = local_client.get_checkpoints()
        if checkpoints and row_filter.delta_after0:
            # No point keeping the delta maps otherwise,
            # we have to poll. If there were no checkpoints, it means
            # we saved without having ever completed a poll.
            #
            # We choose the cp0 as our beginning TID at which to
            # resume polling. We have information on cached data as it
            # relates to those checkpoints. (TODO: Are we sure that
            # the delta maps we've just built are actually accurate
            # as-of this particular TID we're choosing to poll from?)
            #
            # XXX: Now that we're tracking a tid globally we can do much better.
            # XXX: This doesn't correspond to what we actually need anymore.
            self.object_index = _ObjectIndex(
                row_filter.delta_after0.maxValue(),
                None,
                row_filter.delta_after0)
            self.object_index.maps[0].update(row_filter.delta_after1)
        else:
            self.object_index = None


class _PersistentRowFilter(object):

    def __init__(self, adapter, delta_type):
        self.adapter = adapter
        self.delta_after0 = delta_type()
        self.delta_after1 = delta_type()
        self.polled_invalid_oids = OidSet()

    def __str__(self):
        return "<PersistentRowFilter>"

    def __call__(self, checkpoints, row_iter):
        if not checkpoints:
            # Nothing to do except put in correct format, no transforms are possible.
            # XXX: Is there really even any reason to return these? We'll probably
            # never generate keys that match them.
            for row in row_iter:
                yield row[:2], row[2:]
        else:
            delta_after0 = self.delta_after0
            delta_after1 = self.delta_after1
            cp0, cp1 = checkpoints

            # {oid: (state, actual_tid)}
            # This holds things that we're not sure about; we hold onto them
            # and run a big query at the end to determine whether they're still valid or
            # not.
            needs_checked = OidOMap()

            for row in row_iter:
                # Rows are (oid, tid, state, tid), where the two tids
                # are always equal.
                key = row[:2]
                value = row[2:]
                oid = key[0]
                actual_tid = value[1]
                # See __poll_replace_checkpoints() to see how we build
                # the delta maps.
                #
                # We'll poll for changes *after* cp0
                # (because we set that as our current_tid/the
                # storage's prev_polled_tid) and update
                # self._delta_after0, but we won't poll for changes
                # *after* cp1. self._delta_after1 is only ever
                # populated when we shift checkpoints; we assume any
                # changes that happen after that point we catch in an
                # updated self._delta_after0.
                #
                # Also, because we're combining data in the local
                # database from multiple sources, it's *possible* that
                # some old cache had checkpoints that are behind what
                # we're working with now. So we can't actually trust
                # anything that we would put in delta_after1 without
                # validating them. We still return it, but we may take
                # it out of delta_after0 if it turns out to be
                # invalid.

                if actual_tid > cp0:
                    delta_after0[oid] = actual_tid
                elif actual_tid > cp1:
                    delta_after1[oid] = actual_tid
                else:
                    # This is too old and outside our checkpoints for
                    # when something changed. It could be good to have it,
                    # it might be something that doesn't change much.
                    # Unfortunately, we can't just stick it in our fallback
                    # keys (oid, cp0) or (oid, cp1), because it might not be current,
                    # and the storage won't poll this far back.
                    #
                    # The solution is to hold onto it and run a manual poll ourself;
                    # if it's still valid, good. If not, someone should
                    # remove it from the database so we don't keep checking.
                    # We also should only do this poll if we have room in our cache
                    # still (that should rarely be an issue; our db write size
                    # matches our in-memory size except for the first startup after
                    # a reduction in in-memory size.)
                    needs_checked[oid] = value
                    continue
                yield key, value

            # Now validate things that need validated.

            # TODO: Should this be a configurable option, like ZEO's
            # 'drop-rather-invalidate'? So far I haven't seen signs that
            # this will be particularly slow or burdensome.
            self._poll_delta_after1()

            if needs_checked:
                self._poll_old_oids_and_remove(needs_checked)
                for oid, value in iteroiditems(needs_checked):
                    # Anything left is guaranteed to still be at the tid we recorded
                    # for it (except in the event of a concurrent transaction that
                    # changed that object; that should be rare.) So these can go in
                    # our fallback keys.
                    yield (oid, cp0), value

    @log_timed
    def _poll_old_oids_and_remove(self, to_check):
        oids = list(to_check)
        # In local tests, this function executes against PostgreSQL 11 in .78s
        # for 133,002 older OIDs; or, .35s for 57,002 OIDs against MySQL 5.7.
        logger.debug("Polling %d older oids stored in cache", len(oids))
        def poll_old_oids_remove(_conn, cursor):
            return self.adapter.mover.current_object_tids(cursor, oids)
        current_tids_for_oids = self.adapter.connmanager.open_and_call(poll_old_oids_remove)

        for oid in oids:
            if (oid not in current_tids_for_oids
                    or to_check[oid][1] != current_tids_for_oids[oid]):
                del to_check[oid]
                self.polled_invalid_oids.add(oid)

        logger.debug("Polled %d older oids stored in cache; %d survived",
                     len(oids), len(to_check))

    @log_timed
    def _poll_delta_after1(self):
        orig_delta_after1 = self.delta_after1
        oids = list(self.delta_after1)
        # TODO: We have a defined transaction range here that we're concerned
        # about. We might be better off using poller.list_changes(), just like
        # __poll_replace_checkpoints() does.
        logger.debug("Polling %d oids in delta_after1", len(oids))
        def poll_oids_delta1(_conn, cursor):
            return self.adapter.mover.current_object_tids(cursor, oids)
        poll_oids_delta1.transaction_isolation_level = self.adapter.connmanager.isolation_load
        poll_oids_delta1.transaction_read_only = True
        current_tids_for_oids = self.adapter.connmanager.open_and_call(poll_oids_delta1)
        self.delta_after1 = _TransactionRangeObjectIndex(1, None, ())
        self.delta_after1.update(current_tids_for_oids)
        invalid_oids = {
            oid
            for oid, tid in iteroiditems(orig_delta_after1)
            if oid not in self.delta_after1 or self.delta_after1[oid] != tid
        }
        self.polled_invalid_oids.update(invalid_oids)
        logger.debug("Polled %d oids in delta_after1; %d survived",
                     len(oids), len(oids) - len(invalid_oids))

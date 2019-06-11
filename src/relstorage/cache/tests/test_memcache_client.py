##############################################################################
#
# Copyright (c) 2009 Zope Foundation and Contributors.
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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest

from hamcrest import assert_that
from nti.testing.matchers import verifiably_provides

from relstorage.cache.interfaces import IStateCache
from relstorage.tests import TestCase

from . import MockOptions
from . import MockOptionsWithFakeMemcache

class AbstractStateCacheTests(TestCase):
    """
    Tests that all implementations of `IStateCache`
    should be able to pass.
    """

    def getClass(self):
        raise unittest.SkipTest("No implementation defined.")

    Options = MockOptions

    def _makeOne(self, **kw):
        options = self.Options.from_args(**kw)
        inst = self.getClass()(options)
        return inst

    def test_provides(self):
        assert_that(self._makeOne(), verifiably_provides(IStateCache))

    def test_set_multi_and_get_multi(self):
        c = self._makeOne()

        c.set_multi({(0, 0): (b'abc', 0),
                     (0, 1): (b'def', 1),
                     (1, 0): (b'ghi', 0)})
        # Hits on primary key
        self.assertEqual(c(0, 0),
                         (b'abc', 0))
        self.assertEqual(c(1, 0),
                         (b'ghi', 0))
        # Hits on secondary key
        self.assertEqual(c(0, -1, 1),
                         (b'def', 1))
        self.assertEqual(c(1, -1, 0),
                         (b'ghi', 0))

        # And those actually copied the data to the primary key, which
        # is now a hit.
        self.assertEqual(c(0, -1),
                         (b'def', 1))
        self.assertEqual(c(1, -1),
                         (b'ghi', 0))

class MemcacheClientTests(AbstractStateCacheTests):

    def setUp(self):
        from relstorage.tests.fakecache import data
        data.clear()

    tearDown = setUp

    Options = MockOptionsWithFakeMemcache

    def getClass(self):
        from relstorage.cache.storage_cache import MemcacheStateCache
        return MemcacheStateCache.from_options

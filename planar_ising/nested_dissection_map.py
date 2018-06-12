from numba import jitclass
from numba.types import int32


@jitclass([('_in_order_map', int32[:]), ('_in_order_pre_order_mapping', int32[:])])
class NestedDissectionMap:

    def __init__(self, in_order_map, in_order_pre_order_mapping):

        self._in_order_map = in_order_map
        self._in_order_pre_order_mapping = in_order_pre_order_mapping

    @property
    def in_order_map(self):

        return self._in_order_map

    @property
    def in_order_pre_order_mapping(self):

        return self._in_order_pre_order_mapping


nested_dissection_map_nb_type = NestedDissectionMap.class_type.instance_type

# automatically generated by the FlatBuffers compiler, do not modify

# namespace: tflite

import my_flatbuffers as flatbuffers

class SqueezeOptions(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAsSqueezeOptions(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = SqueezeOptions()
        x.Init(buf, n + offset)
        return x

    # SqueezeOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # SqueezeOptions
    def SqueezeDims(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.Get(flatbuffers.number_types.Int32Flags, a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 4))
        return 0

    # SqueezeOptions
    def SqueezeDimsAsNumpy(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.GetVectorAsNumpy(flatbuffers.number_types.Int32Flags, o)
        return 0

    # SqueezeOptions
    def SqueezeDimsLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

def SqueezeOptionsStart(builder): builder.StartObject(1)
def SqueezeOptionsAddSqueezeDims(builder, squeezeDims): builder.PrependUOffsetTRelativeSlot(0, flatbuffers.number_types.UOffsetTFlags.py_type(squeezeDims), 0)
def SqueezeOptionsStartSqueezeDimsVector(builder, numElems): return builder.StartVector(4, numElems, 4)
def SqueezeOptionsEnd(builder): return builder.EndObject()

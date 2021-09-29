# automatically generated by the FlatBuffers compiler, do not modify

# namespace: tflite

import my_flatbuffers as flatbuffers

class UnpackOptions(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAsUnpackOptions(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = UnpackOptions()
        x.Init(buf, n + offset)
        return x

    # UnpackOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # UnpackOptions
    def Num(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

    # UnpackOptions
    def Axis(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

def UnpackOptionsStart(builder): builder.StartObject(2)
def UnpackOptionsAddNum(builder, num): builder.PrependInt32Slot(0, num, 0)
def UnpackOptionsAddAxis(builder, axis): builder.PrependInt32Slot(1, axis, 0)
def UnpackOptionsEnd(builder): return builder.EndObject()

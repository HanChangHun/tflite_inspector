# automatically generated by the FlatBuffers compiler, do not modify

# namespace: tflite

import my_flatbuffers as flatbuffers

class HardSwishOptions(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAsHardSwishOptions(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = HardSwishOptions()
        x.Init(buf, n + offset)
        return x

    # HardSwishOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

def HardSwishOptionsStart(builder): builder.StartObject(0)
def HardSwishOptionsEnd(builder): return builder.EndObject()

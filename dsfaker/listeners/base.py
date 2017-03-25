class Listener:
    def put_single(self, value):
        raise NotImplementedError("put_single not implemented!")

    def put_batch(self, values):
        raise NotImplementedError("put_batch not implemented!")

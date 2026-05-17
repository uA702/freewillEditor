import app.core.effects as fx

class EffectsPipeline:
    def __init__(self):
        self.layers = []  # This list holds our effects in order

    def add_layer(self, effect: fx.baseEffect):
        self.layers.append(effect)

    def process(self, frame):
        for layer in self.layers:
            frame = layer.apply(frame)
        return frame
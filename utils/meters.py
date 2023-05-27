class Meters(object):
    # Calculem contadors de valors i temporitzadors
    def __init__(self):
        self.reset()

    def reset(self):
        # Reiniciem valors
        self.count = 0
        self.sum = 0
        self.avg = 0
        self.val = 0

    def update(self, val, num=1):
        # Actualitzem
        self.count += num
        self.sum += val * num
        self.avg = self.sum / self.count
        self.val = val
class Parameter:
    def __init__(self, k, w, percentile, boundary_diff, np_percent, cue_percent, precision, recall, f1):
        self.k = k
        self.w = w
        self.percentile = percentile
        self.boundary_diff = boundary_diff
        self.np_percent = np_percent
        self.cue_percent = cue_percent
        self.precision = precision
        self.recall = recall
        self.f1 = f1
    def __repr__(self):
        return repr((self.w, self.k, self.percentile, self.boundary_diff, self.np_percent, self.cue_percent, self.precision, self.recall, self.f1))
from Model import Model


class ModulatedMultidimensionalAtmosphericScrubbers(QCAlgorithm):

    def Initialize(self):
        self.SetStartDate(2015, 8, 15)  # Set Start Date
        self.SetEndDate(2020, 8, 15)
        self.SetCash(100000)  # Set Strategy Cash
        self.SetBenchmark('SPY')

        self.asset1 = self.AddEquity('GLD', Resolution.Daily).Symbol
        self.asset2 = self.AddEquity('SLV', Resolution.Daily).Symbol
        self.SetWarmup(200)
        self.model = Model()

        # retrain our model periodically
        self.Train(self.DateRules.MonthStart('GLD'), self.TimeRules.Midnight, self.TrainModel)
        self.months = 0

    def OnData(self, data):
        self.model.Update(self.Time, data[self.asset1].Close, data[self.asset2].Close)

        if not self.Portfolio.Invested and self.model.EnterTrade():
            self.SetHoldings(self.A, 1)
            self.SetHoldings(self.B, -self.model.alpha)
            self.entrytime = self.Time
        elif self.Portfolio.Invested and self.model.ExitTrade(self.Time,self.entrytime):
            self.Liquidate()

    def TrainModel(self):

        self.months += 1

        # only retrain every 7 months
        if self.months % 7 != 1:
            return

        self.model.Train()

        theta = self.model.theta
        mu = self.model.mu
        sigma = self.model.sigma
        self.Log('θ: ' + str(round(theta, 2)) + ' μ: ' + str(round(mu, 2)) + ' σ: ' + str(round(sigma, 2)))
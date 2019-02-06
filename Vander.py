
function y = vandermonde1(x)
  n = length(x)
  for i = 1:n
    y(:, i) = x^(i-1)
  end
endfunction

function y = vandermonde2(a)
  n = length(a)
  y=[a*ones(1,4)].^[[0:n-1]'*ones(1,4)]'
endfunction

function y = vandermonde3(a)
  n = length(a)
  y = kron(a, ones(1, n)).^kron([0:n-1]', ones(1, n))'
endfunction

from base import Algorithm, mean

class LinregNonMatrix(Algorithm):

    def train(self, x, y):

        x_mean = mean(x)

        y_mean = mean(y)

        x_dev = sum([i-x_mean for i in x])

        y_dev = sum([i-y_mean for i in y])

        self.slope = (x_dev*y_dev)/(x_dev*x_dev)

        self.intercept = y_mean - (self.slope*x_mean)

    def predict(self, x):

        return [i*self.slope + self.intercept for i in x]


class LinregListMatrix(Algorithm):

    def train(self, X, y):

        pass

#!/usr/bin/env python3

import json
import matplotlib.pyplot as plt
import numpy as np
import random
import sys
import tqdm

NUM_RUNS = 10

### Problem definition

SWITCHING_COST = 1 - 0.0001  # adding a small noise to make sure all algorithms break ties in a consistent way
VANILLA_COST = 1
CHOCOLATE_COST = 2
MISMATCH_FACTOR = 2

### Utils

def EvaluateSolutionStepByStep(requests, solution):
  assert len(requests) == len(solution)
  assert all(elem in 'VC' for elem in solution)
  service_cost = [
    (VANILLA_COST if r == 'V' else CHOCOLATE_COST) * (MISMATCH_FACTOR if r != s else 1)
    for r,s in zip(requests, solution)]
  switching_cost = [0] + [
    (SWITCHING_COST if prv != nxt else 0)
    for prv, nxt in zip(solution[:-1], solution[1:])]
  cost = [a + b for a, b in zip(service_cost, switching_cost)]
  return cost


def EvaluateSolution(requests, solution):
  return sum(EvaluateSolutionStepByStep(requests, solution))

### Algorithms

def Random(requests):
  return random.choices('VC', k=len(requests))


def LazyRandom(requests):
  state = requests[0]
  solution = []
  for request in requests:
    if request != state and random.choice((True, False)):
      state = request
    solution.append(state)
  return solution


def Greedy(requests):
  return requests


def ComputeDP(requests):
  DP = [(0,0),]
  for request in requests:
    PREV = DP[-1][::]
    if request == 'V':
      DP.append((
        VANILLA_COST + min(PREV[0], PREV[1] + SWITCHING_COST),
        VANILLA_COST * MISMATCH_FACTOR + min(PREV[0] + SWITCHING_COST, PREV[1])))
    else:
      DP.append((
        CHOCOLATE_COST * MISMATCH_FACTOR + min(PREV[0], PREV[1] + SWITCHING_COST),
        CHOCOLATE_COST + min(PREV[0] + SWITCHING_COST, PREV[1])))
  return DP


def Opt(requests, error_probability=0.0):
  DP = ComputeDP(requests[::-1])[::-1]
  solution = []
  j = DP[0].index(min(DP[0]))
  for i in range(len(requests)):
    request_i = 'VC'.index(requests[i])
    if DP[i][1 - j] + SWITCHING_COST < DP[i][j]:
      j = 1 - j
    if random.random() < error_probability:
      j = 1 - j
    solution.append('VC'[j])
  assert error_probability > 0 or round(EvaluateSolution(requests, solution)) == round(min(DP[0]))
  return solution


def FtP(requests, predictions):
  solution = []
  for request, prediction in zip(requests, predictions):
    if request == 'C' and prediction == 'V':
      solution.append('C')
    else:
      solution.append(prediction)
  return solution


def WorkFunction(requests):
  DP = ComputeDP(requests)
  solution = []
  current_state = 0
  for i, request in enumerate(requests):
    stay_cost = DP[i + 1][current_state]
    switch_cost = DP[i + 1][1 - current_state] + SWITCHING_COST
    if switch_cost < stay_cost:
      current_state = 1 - current_state
    solution.append('VC'[current_state])
  return solution

### Combining algorithms

def CombineDeterministic(requests, solutions, gamma=0.01):
  costs = [np.cumsum(EvaluateSolutionStepByStep(requests, solution)) for solution in solutions]
  n_algorithms = len(solutions)
  assert n_algorithms >= 2
  current_algorithm = 0
  max_cost = 1
  solution = []
  for i in range(len(requests)):
    while (costs[current_algorithm][i] > max_cost):
      current_algorithm = (current_algorithm + 1) % n_algorithms
      max_cost *= (1 + gamma / (n_algorithms - 1))
    solution.append(solutions[current_algorithm][i])
  return solution


def CombineRandomized(requests, solutions, epsilon=0.5):

  beta = 1.0 - 0.5 * epsilon
  assert len(solutions) == 2  # we don't implement combining more than two algorithms
  costs = [EvaluateSolutionStepByStep(requests, solution) for solution in solutions]
  weights = (0.5, 0.5)
  current_algorithm = 0
  solution = []
  for i in range(len(requests)):
    new_weights = tuple(weight * (beta) ** costs[i] for weight, costs in zip(weights, costs))
    total = sum(new_weights)
    new_weights = tuple(weight / total for weight in new_weights)
    
    if new_weights[current_algorithm] < weights[current_algorithm]:
      if random.random() < (weights[current_algorithm] - new_weights[current_algorithm]) / weights[current_algorithm]:
        current_algorithm = 1 - current_algorithm
    solution.append(solutions[current_algorithm][i])
    weights = new_weights    
  return solution

### Main

def main():
  ERROR_PROBABILITIES = np.linspace(0., .5, 11)
  ALGORITHMS_ONLINE = (
    ('OPT', Opt),
    # ('SilyRandom', Random),
    # ('Random', LazyRandom),
    # ('Greedy (1.25-competitive)', Greedy),
    ('Work Function', WorkFunction),
  )
  ALGORITHMS_WITH_PREDICTIONS = (
    ('FtP', lambda requests, predictions: predictions),
    ('RobustFtP deterministic', lambda requests, predictions: CombineDeterministic(requests, (predictions, WorkFunction(requests)))),
    ('RobustFtP randomized', lambda requests, predictions: CombineRandomized(requests, (predictions, WorkFunction(requests)))),
#     ('Deterministic (gamma=0.1)', lambda requests, predictions: CombineDeterministic(requests, (predictions, WorkFunction(requests)), gamma=0.1)),
#     ('Deterministic (gamma=0.01)', lambda requests, predictions: CombineDeterministic(requests, (predictions, WorkFunction(requests)), gamma=0.01)),
#     ('Deterministic (gamma=0.001)', lambda requests, predictions: CombineDeterministic(requests, (predictions, WorkFunction(requests)), gamma=0.001)),
#     ('Randomized (epsilon=0.5)', lambda requests, predictions: CombineRandomized(requests, (predictions, WorkFunction(requests)), epsilon=0.5)),
#     ('Randomized (epsilon=0.3)', lambda requests, predictions: CombineRandomized(requests, (predictions, WorkFunction(requests)), epsilon=0.3)),
#     ('Randomized (epsilon=0.1)', lambda requests, predictions: CombineRandomized(requests, (predictions, WorkFunction(requests)), epsilon=0.1)),
#     ('Randomized (epsilon=0.01)', lambda requests, predictions: CombineRandomized(requests, (predictions, WorkFunction(requests)), epsilon=0.01)),
#     ('Randomized (epsilon=0.001)', lambda requests, predictions: CombineRandomized(requests, (predictions, WorkFunction(requests)), epsilon=0.001)),
    # ('Everything', lambda requests, predictions: CombineDeterministic(requests, (predictions, WorkFunction(requests), CombineRandomized(requests, (predictions, WorkFunction(requests)))))),
  )
  ALGORITHMS = ALGORITHMS_ONLINE + ALGORITHMS_WITH_PREDICTIONS

  runs = []
  for run in range(NUM_RUNS):
    print('Run %d out of %d' % (run + 1, NUM_RUNS))
    costs = [[0 for _ in ERROR_PROBABILITIES] for _ in ALGORITHMS]

    for dataset in tqdm.tqdm(sys.argv[1:]):
      with open(dataset) as f:
        requests = tuple(line.rstrip('\n') for line in f)
      # requests = random.choices('VC', k=1000)  # use synthetic random dataset instead
      for k, (_, algorithm) in enumerate(ALGORITHMS_ONLINE):
        costs[k][0] += EvaluateSolution(requests, algorithm(requests))
      for i, p in enumerate(ERROR_PROBABILITIES):
        predictions = FtP(requests, Opt(requests, p))
        for k, (_, algorithm) in enumerate(ALGORITHMS_WITH_PREDICTIONS):
          costs[k + len(ALGORITHMS_ONLINE)][i] += EvaluateSolution(requests, algorithm(requests, predictions))

    for k in range(len(ALGORITHMS_ONLINE)):
      costs[k] = [costs[k][0],] * len(ERROR_PROBABILITIES)

    runs.append(costs)

  print("Max stddev: %f" % np.max(np.std(np.array(runs) / runs[0][0][0], axis=0)))

  costs = np.mean(runs, axis=0)

  plt.rcParams.update({
    'pgf.texsystem': 'pdflatex',
    'font.family': 'serif',
    'font.size': 9,
    'pgf.rcfonts': False,
  })

  LINES = ['-', '--', '-.', ':']
  for k, (label, algorithm) in list(enumerate(ALGORITHMS))[1:]:  # ignore OPT
    plt.plot(ERROR_PROBABILITIES, [x / costs[0][0] for x in costs[k]], label=label, ls=LINES[(k - 1) % len(LINES)])

  xlabel = plt.xlabel('Error probability of the synthetic predictor')
  ylabel = plt.ylabel('Competitive ratio')
  plt.legend(loc='upper left')
  plt.ylim(0.999, 1.109)
  plt.savefig('icecream.png', dpi=150)
  plt.gcf().set_size_inches(w=3.25, h=2.25)
  plt.savefig('icecream.pgf', bbox_extra_artists=[xlabel, ylabel], bbox_inches='tight')
  plt.show()


if __name__ == '__main__':
  main()

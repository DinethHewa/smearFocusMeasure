# GP Hyperparameter Manuscript Text

The saved production GP runs used population size 500, 100 generations, tournament size 3, crossover probability 0.5, mutation probability 0.2, maximum tree depth 10, elitism 1, and NSGA-II = True. The primary objective was corrected_generalization_score and the secondary objective was expression_complexity.

Recorded audit status: core hyperparameters were consistent across saved runs. Fields added in later resume/run-control patches, including max_nodes, max_eval_seconds, and device, are not present in every older LODO result file; where recorded, the dominant values were max_nodes = 35, max_eval_seconds = 30.0, and device = cpu.

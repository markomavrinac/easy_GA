#dizajnirati shell algoritma slican onom
#kromosom se posebno dizajnira i samo hook upa na objekt algoritma
#fitness function mora kao argument uzimati samo kromosom i returnati fitnessscore
import random
import string

"""
chromosome - object of a custom class, the reason for this is so some additional stuff related to fitness function may be designed
params - a dict containing GA parameters displayed in the 'defaults' variable
args - tuple('int'/'float', list of tuples defining limits for chromosome parameters), tuple('string', number_of_args), tuple('bool', number_of_args)
fitness_function - handle to a fitness function that needs to be a class function of chromosome object's class

"""
class GeneticAlgorithm(object):
    def __init__(self, params, fitness_function, chromosome, args, arg_names = None):
        self.defaults = {
        'pop_size' : 15,
        'relative_tolerance' : 10**-5, #if difference between two gen is smaller than this parameter, the algorithm stops
        'absolute_tolerance' : False, #if gen fitness score is bigger than this, algorithm stops
        'selection' : 'no_selection', #best members still get to the next generation, but all are favored instead of just top N determined by elitism
        'elitism_threshold': False, #how many of the best chromosomes get to be kept in the population
        'elitism_trigger': False, #when average generation fitness reaches float value for triggers, it switches to another selection
        'ranked_trigger' : False, 
        'ranked_rank_probability': [0.9,0.6,0.5,0.3], #splits population into len(arg) parts, then assigns each rank probability equal to index list member
        'proportional_trigger': False,
        'mutation_rate' : 0.2,
        'mutation_repeats' : 1,
        'mutation_intensity' : 10, #for float/int representation
        'duplicates' : False,
        'max_iter' : 50,
        'seed' : False,
        'prop_trig' : False,
        'ranked_trig' : False,
        'char_list' : f' {string.ascii_letters}{string.digits}{string.punctuation}',
        'keep_parents_during_crossover' : True,
        'number_of_safe_gens' : 5, #number of gens before stop conditions start to apply
        'filter_return': False, #if this value is true, GA only returns chromosomes >= of 'filter_threshold'
        'filter_threshold': 1.0,
        'best_member_fitness_stop_condition':None
        }
        self.chromosome_repr = args[0]
        self.population = {} #entire population -> key = gen | value(list) = chromosomes
        self.settings = {}
        self.__set_params(params)
        self.chromosome = chromosome #reference to Chromosome class
        self.fitness_function = fitness_function #reference to fitness function
        self.args = args
        self.arg_names = arg_names
        self.calculated = {}
        self.best_fitness = {} #fitness of the best chromosome in per generation, key = gen - value = fitness score
        self.average_fitness = {} #same but average of each generation
        
    def __set_params(self, params):
        for param in self.defaults:
            if param in params:
                self.settings[param] = params[param]
                if not type(self.settings[param])==type(self.defaults[param]):
                    if param!='elitism_threshold' and param!='ranked_trigger' and param!='elitism_trigger' and param!='proportional_trigger' and param!='best_member_fitness_stop_condition' and param!='filter_threshold':
                        raise TypeError(f"Parameter {param} must be {type(self.defaults[param])}, not {type(self.settings[param])}!")
            else:
                self.settings[param] = self.defaults[param]
    def __get_stats(self, gen):
        keys = list(self.population[gen].keys())
        self.best_fitness[gen] = max(self.population[gen].values())
        self.average_fitness[gen] = self.__get_generation_avg(gen)
    def __create_pop(self): #initialize population
        self.population = {} #reset in case of a rerun
        if self.chromosome_repr == 'int':
            self.population[0] = self.__create_pop_integer()
        elif self.chromosome_repr == 'float':
            self.population[0] = self.__create_pop_float()
        elif self.chromosome_repr == 'string':
            self.population[0] = self.__create_pop_string()
        elif self.chromosome_repr == 'bool':
            self.population[0] = self.__create_pop_bool()
        
        
        self.__sort_gen(0)
        self.__get_stats(0)
        
    def __round_up_div(self, value1, value2): #for ranked member division
        return int(value1//value2) + (value1 % value2 > 0)
        
    def __selection(self, raw_population): #raw_population -> dict containing a population "chromosome" : fitness_score
        selected = {}
        if self.settings['selection']=='no_selection':
            return raw_population
        elif self.settings['selection']=='elitism':
            counter = 0
            for chromosome in raw_population:
                if counter<self.settings['elitism_threshold']:
                    selected[chromosome] = raw_population[chromosome]
                else:
                    break
                counter+=1
        elif self.settings['selection']=='ranked':
            nr_of_ranks = len(self.settings['ranked_rank_probability'])
            ranks = dict(zip(self.settings['ranked_rank_probability'], nr_of_ranks*[])) # dict{probability : chromosome list}
            rank_step = self.__round_up_div(self.settings['pop_size'],nr_of_ranks) #how many chromosomes per rank, it's a bit approximate
            counter=0 #counter to check when index_counter needs to be bumped
            index_counter = 0 #counter for rank probability list
            for chromosome in raw_population:
                if counter==rank_step:
                    index_counter+=1
                    counter=0
                counter+=1
                if random.random()<=self.settings['ranked_rank_probability'][index_counter]:
                    selected[chromosome] = self.__get_fitness(chromosome)           
        elif self.settings['selection']=='proportional': #this is assuming fitness score goes from 0-1, so if you're going to use it normalize the score (score=currentScore/maxScore) otherwise it will behave as if there was no selection
            for chromosome in raw_population:
                if random.random()<=self.__get_fitness(chromosome):
                    selected[chromosome] = self.__get_fitness(chromosome)
        return selected
    def __toggle_selection_mode(self, gen): #switch modes based on triggers
        if self.settings['elitism_trigger']:
            if self.__get_generation_avg(gen)>=self.settings['elitism_trigger']:
                self.settings['selection'] = 'elitism'
                self.settings['elitism_trigger'] = False
        if self.settings['ranked_trigger']:
            if self.__get_generation_avg(gen)>=self.settings['ranked_trigger']:
                self.settings['selection'] = 'ranked'
                self.settings['ranked_trigger'] = False
        if self.settings['proportional_trigger']:
            if self.__get_generation_avg(gen)>=self.settings['proportional_trigger']:
                self.settings['selection'] = 'proportional'
                self.settings['proportional_trigger'] = False
    def __create_pop_integer(self):
        population = {}
        while len(population)<self.settings['pop_size']:
            args = []  
            for arg_range in self.args[1]:
                arg = random.randrange(arg_range[0], arg_range[1]+1)
                args.append(arg)
            chromosome = self.chromosome(args, self.arg_names, self.chromosome_repr, limits = self.args[1])
            population[chromosome] = self.__get_fitness(chromosome)
        return population
    def __create_pop_float(self):
        population = {}
        while len(population)<self.settings['pop_size']:
            args = []  
            for arg_range in self.args[1]:
                arg = (arg_range[1]-arg_range[0])*random.random()+arg_range[0]
                args.append(arg)
            chromosome = self.chromosome(args, self.arg_names, self.chromosome_repr, limits = self.args[1])
            population[chromosome] = self.__get_fitness(chromosome)
        return population
    def __create_pop_string(self):
        population = {}
        while len(population)<self.settings['pop_size']:
            args = []  
            for arg_index in range(self.args[1]):
                arg = self.settings['char_list'][random.randrange(len(self.settings['char_list']))]
                args.append(arg)
            chromosome = self.chromosome(args, self.arg_names, self.chromosome_repr, self.settings['char_list'])
            population[chromosome] = self.__get_fitness(chromosome)
        return population
    def __create_pop_bool(self):
        population = {}
        while len(population)<self.settings['pop_size']:
            args = []  
            for arg_index in range(self.args[1]):
                arg = random.randrange(2)
                args.append(arg)
            chromosome = self.chromosome(args, self.arg_names, self.chromosome_repr)
            population[chromosome] = self.__get_fitness(chromosome)
        return population    
    def __get_fitness(self, chromosome):
        if chromosome.get_args() in self.calculated:
            fitness_score = chromosome.set_fitness(self.calculated[chromosome.get_args()]) #taking return from set_fitness
        else:
            fitness_score = self.fitness_function(chromosome)
            self.calculated[chromosome.get_args()] = fitness_score
        return fitness_score
        
    def __next_gen(self, gen): #create next generation
        new_generation = {}
        crossover_selected = self.__selection(self.population[gen-1])
        new_gen_raw = self.__crossover_gen(crossover_selected)
        sorted_gen = sorted(new_gen_raw, key=self.fitness_function)
        sorted_gen.reverse()
        for chromosome in sorted_gen:
            if len(new_generation)<self.settings['pop_size']:
                new_generation[chromosome] = self.__get_fitness(chromosome)
            else:
                break
        self.population[gen] = new_generation
        self.__get_stats(gen)
        self.__sort_gen(gen)
        self.__toggle_selection_mode(gen)
    def __mutate(self, chromosome): #mutate a chromosome
        pass
    def __crossover_gen(self, population):
        new_gen_candidates = {}
        dupes = []
        keys = list(population.keys())
        for parent_index1 in range(len(keys)):
            for parent_index2 in range(parent_index1+1, len(keys)):
                children = self.__crossover(keys[parent_index1], keys[parent_index2])
                for child in children:
                    if child.get_args() not in dupes:
                        new_gen_candidates[child] = self.__get_fitness(child)
                        dupes.append(child.get_args())
        if self.settings['keep_parents_during_crossover']:
            for chromosome in population:
                if chromosome.get_args() not in dupes:
                    new_gen_candidates[chromosome] = self.__get_fitness(chromosome)
                    dupes.append(chromosome.get_args())
        return new_gen_candidates
                
    def __crossover(self, parent1, parent2): #crossover (two chromosomes) parent1, parent2 -> chromosome object
        args1 = parent1.get_values()
        args2 = parent2.get_values()
        if len(args1)<=len(args2):
            cross_point = random.randrange(1, len(args1))
        else:
            cross_point = random.randrange(1, len(args2))
        if self.args[0]=='string':
            child1 = self.chromosome(args1[:cross_point]+args2[cross_point:], self.arg_names, self.chromosome_repr, limits = self.settings['char_list'])
            child2 = self.chromosome(args2[:cross_point]+args1[cross_point:], self.arg_names, self.chromosome_repr, limits = self.settings['char_list'])
        else:
            child1 = self.chromosome(args1[:cross_point]+args2[cross_point:], self.arg_names, self.chromosome_repr, limits = self.args[1])
            child2 = self.chromosome(args2[:cross_point]+args1[cross_point:], self.arg_names, self.chromosome_repr, limits = self.args[1])
        child1.mutate(self.settings['mutation_rate'], self.settings['mutation_repeats'], self.settings['mutation_intensity'])
        child2.mutate(self.settings['mutation_rate'], self.settings['mutation_repeats'], self.settings['mutation_intensity'])
        return [child1,child2]
    def __sort_gen(self, gen):
        sorted_gen = sorted(self.population[gen], key=self.fitness_function)
        sorted_gen.reverse()
        sorted_fitness = []
        for chromosome in sorted_gen:
            sorted_fitness.append(self.__get_fitness(chromosome))
        sorted_gen_dict = dict(zip(sorted_gen,sorted_fitness))
        self.population[gen] = sorted_gen_dict
        
    def __remove_dupes(self, gen):
        pass
    def __stop_conditions(self, gen):
        if gen>self.settings['number_of_safe_gens']:
            if abs(self.__get_generation_avg(gen-1)-self.__get_generation_avg(gen))<self.settings['relative_tolerance']: #difference between two gens
                return True
            elif self.settings['absolute_tolerance']:
                if self.__get_generation_avg(gen)>=self.settings['absolute_tolerance']:
                    return True
            elif type(self.settings['best_member_fitness_stop_condition'])==float:
                if self.best_fitness[gen]>=self.settings['best_member_fitness_stop_condition']:
                    return True
        return False
    def __get_generation_avg(self, gen):
        sum = 0.0
        for chromosome in self.population[gen]:
            sum+=self.__get_fitness(chromosome)
        try:
            return sum/len(self.population[gen])
        except ZeroDivisionError:
            return 0
    def __display_gen(self, gen):
        print(f"Gen {gen}: Average fitness -> {self.__get_generation_avg(gen)}")
        for chromosome in self.population[gen]:
            print(f"{chromosome.get_args()} -> {self.population[gen][chromosome]}")
            
    def start(self, verbose = False):
        self.__create_pop()
        if verbose:
            print(f"Initialize population! Fitness = {self.__get_generation_avg(0)}")        
        for gen in range(1, self.settings['max_iter']):
            self.__next_gen(gen)
            if verbose:
                print(f"Generation {gen} -> Fitness = {self.__get_generation_avg(gen)}")
            if self.__stop_conditions(gen):
                if verbose:
                    print(f"End")
                break
        results = []
        for chromosome in self.population[gen]:
            if self.settings['filter_return']:
                if self.__get_fitness(chromosome)>=self.settings['filter_threshold']:
                    results.append((chromosome.get_values(),self.population[gen][chromosome]))
            else:
                results.append((chromosome.get_values(),self.population[gen][chromosome]))
        return [results, (self.best_fitness, self.average_fitness)]

"""
args - a list containing all chromosome arguments/parameters (unlike in GA class where it's a tuple ranging from-to)
argnames - give each argument a name
representation - string type object defining representation, defined within GA class
"""
class Chromosome(object):
    def __init__(self, args, argnames, representation, limits = None):
        try:
            self.params = dict(zip(argnames, args))
        except:
            self.params = dict(zip(range(len(args)), args))
        self.fitness_score = False
        self.representation = representation
        self.limits = limits
    def get_fitness(self):
        if not self.fitness_score:
            raise Exception("Fitness score not calculated!")
        else:
            return self.fitness_score
    def __limit(self, n, index):
        if self.representation=='int':
            return min(max(n, self.limits[index][0]), self.limits[index][1])
        elif self.representation=='float':
            return float(min(max(n, self.limits[index][0]), self.limits[index][1]))
        
    def mutate(self, rate, repeats = 1, intensity=None): #limits: for int and float [minrange, maxrange], string pulls limits out of self.settings['char_list']
        #mutate part
        if random.random()<rate:
            for i in range(repeats):
                key = random.choice(list(self.params.keys()))
                if self.representation == 'int': 
                    key_index = list(self.params).index(key)
                    self.params[key] = self.__limit(self.params[key] + random.randrange(-intensity,intensity+1), key_index)
                elif self.representation == 'float':
                    key_index = list(self.params).index(key)                
                    self.params[key] = self.__limit(self.params[key] + 2*intensity*random.random()-intensity, key_index)
                elif self.representation == 'string':
                    self.params[key] = random.choice(self.limits)
                elif self.representation == 'bool':
                    self.params[key] = 1- self.params[key]
    def set_fitness(self, value):
        self.fitness_score = value
        return self.fitness_score
    def get_args(self):
        return f"{self.params}"
    def get_values(self):
        return list(self.params.values())
    def __str__(self):
        return str(list(self.params.values()))
#ga1 = GeneticAlgorithm({'pop_size':22}, Chromosome, ('int',3*[(0,15),(0,7)]), ["Kp1", "Ki1", "Kp2", "Ki2", "Kp3", "Ki3"], Chromosome.get_fitness)
#ga1._create_pop()




params = {
    'pop_size' : 15,
    'tolerance' : 0.0001,
    'selection' : 'elitism',
    'mutation_rate' : 0.2,
    'mutation_intensity' : 10,
    'duplicates' : False,
    'max_iter' : 200,
    'seed' : False,
    'prop_trig' : False,
    'ranked_trig':False
    }
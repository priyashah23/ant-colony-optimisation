def selection_menu():
    answer = 0
    while answer not in {1, 2}:
        answer = int(input("Select 1: Burma and Select 2: Brazil\n"))
    return answer


def select_colony_variation() -> int:
    answer = 0
    colony_variation_choice = [10, 50, 100]
    while answer not in {1, 2, 3}:
        answer = int(input("Select Colony Size: \n1: 10, \n2: 50, \n3: 100\n"))
    return colony_variation_choice[answer - 1]


def select_decay_factor() -> float:
    answer = 0
    decay_factor_choice = [0.1, 0.5, 0.9]
    while answer not in {1, 2, 3}:
        answer = int(input("Select Decay Factor: \n1: 0.1, \n2: 0.5, \n3: 0.9\n"))
    return decay_factor_choice[answer - 1]


def select_alpha() -> float:
    answer = 0
    alpha_choice = [0.5, 1, 3]
    while answer not in {1, 2, 3}:
        answer = int(input("Select what constant Alpha should be: \n1: 0.5, \n2: 1, \n3: 3\n"))
    return alpha_choice[answer - 1]


def select_beta() -> float:
    answer = 0
    beta_choice = [1, 3, 5]
    while answer not in {1, 2, 3}:
        answer = int(input("Select what constant Beta should be: \n1: 1, \n2: 3, \n3: 5\n"))
    return beta_choice[answer - 1]


def select_parameters():
    colony = select_colony_variation()
    decay_factor = select_decay_factor()
    alpha = select_alpha()
    beta = select_beta()
    return [decay_factor, colony, alpha, beta]

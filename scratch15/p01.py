from scratch15.ex02_wo import Split, Leaf


class Candidate(NamedTuple):
    '''NamedTuple을 상속받는 클래스 선언'''
    level: str
    lang: str
    tweets: bool
    phd: bool
    result: bool = None




if __name__ == '__main__':
    candidates = [Candidate('Senior', 'Java', False, False, False),
                  Candidate('Senior', 'Java', False, True, False),
                  Candidate('Mid', 'Python', False, False, True),
                  Candidate('Junior', 'Python', False, False, True),
                  Candidate('Junior', 'R', True, False, True),
                  Candidate('Junior', 'R', True, True, False),
                  Candidate('Mid', 'R', True, True, True),
                  Candidate('Senior', 'Python', False, False, False),
                  Candidate('Senior', 'R', True, False, True),
                  Candidate('junior', 'Python', True, False, True),
                  Candidate('Senior', 'Python', True, True, True),
                  Candidate('Mid', 'Python', False, True, True),
                  Candidate('Mid', 'Java', True, False, True),
                  Candidate('Junior', 'Python', False, True, False)]



# 왼쪽 칠판 기준 hire_tree


hire_tree = Split('lang',
                # sub-tree
                  {'Java': Split('level',
                                        {'Senior': Leaf(False),
                                            'Mid': Leaf(True)}),

                   'Python': Split('level',
                                        {'Senior':
                                            Split('tweets', {True: Leaf(True), False: Leaf(False)}),
                                            'Mid':
                                            Split('phd', {True: Leaf(True), False: Leaf(False)}),
                                         'Junior':
                                            Split('tweets', {True: Leaf(True),
                                                             False: Split('phd', {True: Leaf(False), False: Leaf(True)
                                               })})}),

                    'R': Split('level',
                                    {'Senior': Leaf(True),
                                        'Mid': Leaf(True),
                                     'Junior': Split('phd', {True: Leaf(False), False: Leaf(True)})}
                            )})
import itertools
from graph_completion import graph_completion


if __name__ == '__main__':
    # graph_completion.main()
    a = list(range(4))
    p = list(itertools.combinations(a, 2))
    print(p)
    print(type(p[0]))
    
    # print(['	'])
    
    # alphbet = list(chr(w) for w in range(ord('a'), ord('z')+1))

    # print(alphbet)
    
    # print(mapping)

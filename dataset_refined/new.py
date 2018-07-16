##########################################
#         Just for Interview purpose     #
##########################################

# Recursion to compute permutation of a number
def perm1(lst):
    l = []
    for i in range(len(lst)):
        x = lst[i]
        xs = lst[:i] + lst[i+1:]
        for p in perm1(xs):
            l.append([x]+p)
    return l

data = list('abc')
print('perm1')
for p in perm1(data):
    print(p)

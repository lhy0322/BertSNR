import sys, string, random

def computeCountAndLists(s):
    # Initialize lists and mono- and dinucleotide dictionaries
    List = {}  # List is a dictionary of lists
    List['A'] = [];
    List['C'] = [];
    List['G'] = [];
    List['T'] = [];
    # FIXME: is this ok?
    List['N'] = []
    nuclList = ["A", "C", "G", "T", "N"]
    s = s.upper()
    # s       = s.replace("U","T")
    nuclCnt = {}  # empty dictionary
    dinuclCnt = {}  # empty dictionary
    for x in nuclList:
        nuclCnt[x] = 0
        dinuclCnt[x] = {}
        for y in nuclList:
            dinuclCnt[x][y] = 0

    # Compute count and lists
    nuclCnt[s[0]] = 1
    nuclTotal = 1
    dinuclTotal = 0
    for i in range(len(s) - 1):
        x = s[i];
        y = s[i + 1]
        List[x].append(y)
        nuclCnt[y] += 1;
        nuclTotal += 1
        dinuclCnt[x][y] += 1;
        dinuclTotal += 1
    assert (nuclTotal == len(s))
    assert (dinuclTotal == len(s) - 1)
    return nuclCnt, dinuclCnt, List


def chooseEdge(x, dinuclCnt):
    z = random.random()
    denom = dinuclCnt[x]['A'] + dinuclCnt[x]['C'] + dinuclCnt[x]['G'] + dinuclCnt[x]['T'] + dinuclCnt[x]['N']
    numerator = dinuclCnt[x]['A']
    if z < float(numerator) / float(denom):
        dinuclCnt[x]['A'] -= 1
        return 'A'
    numerator += dinuclCnt[x]['C']
    if z < float(numerator) / float(denom):
        dinuclCnt[x]['C'] -= 1
        return 'C'
    numerator += dinuclCnt[x]['G']
    if z < float(numerator) / float(denom):
        dinuclCnt[x]['G'] -= 1
        return 'G'
    numerator += dinuclCnt[x]['T']
    if z < float(numerator) / float(denom):
        dinuclCnt[x]['T'] -= 1
        return 'T'
    dinuclCnt[x]['N'] -= 1
    return 'N'


def connectedToLast(edgeList, nuclList, lastCh):
    D = {}
    for x in nuclList: D[x] = 0
    for edge in edgeList:
        a = edge[0];
        b = edge[1]
        if b == lastCh: D[a] = 1
    for i in range(3):
        for edge in edgeList:
            a = edge[0];
            b = edge[1]
            if D[b] == 1: D[a] = 1
    ok = 0
    for x in nuclList:
        if x != lastCh and D[x] == 0: return 0
    return 1


def eulerian(s):
    nuclCnt, dinuclCnt, List = computeCountAndLists(s)
    # compute nucleotides appearing in s
    nuclList = []
    for x in ["A", "C", "G", "T", "N"]:
        if x in s: nuclList.append(x)
    # create dinucleotide shuffle L
    firstCh = s[0]  # start with first letter of s
    lastCh = s[-1]
    edgeList = []
    for x in nuclList:
        if x != lastCh: edgeList.append([x, chooseEdge(x, dinuclCnt)])
    ok = connectedToLast(edgeList, nuclList, lastCh)
    return ok, edgeList, nuclList, lastCh


def shuffleEdgeList(L):
    n = len(L);
    barrier = n
    for i in range(n - 1):
        z = int(random.random() * barrier)
        tmp = L[z]
        L[z] = L[barrier - 1]
        L[barrier - 1] = tmp
        barrier -= 1
    return L


def dinuclShuffle(s):
    # Significance of nucleotide sequence alignments: a method for random sequence permutation that preserves
    # dinucleotide and codon usage
    ok = 0
    while not ok:
        ok, edgeList, nuclList, lastCh = eulerian(s)
    nuclCnt, dinuclCnt, List = computeCountAndLists(s)

    # remove last edges from each vertex list, shuffle, then add back
    # the removed edges at end of vertex lists.
    for [x, y] in edgeList: List[x].remove(y)
    for x in nuclList: shuffleEdgeList(List[x])
    for [x, y] in edgeList: List[x].append(y)

    # construct the eulerian path
    L = [s[0]];
    prevCh = s[0]
    for i in range(len(s) - 2):
        ch = List[prevCh][0]
        L.append(ch)
        del List[prevCh][0]
        prevCh = ch
    L.append(s[-1])
    t = "".join(L)
    return t

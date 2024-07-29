
def logargs(args, width=120):
    length = 1
    L=[]
    l= "|"
    for id,arg in enumerate(vars(args)):
        name,value = arg, str(getattr(args, arg))
        nv = name+":"+value
        if length +(len(nv)+2)>width:
            L.append(l)
            l = "|"
            length = 1
        l += nv + " |"
        length += (len(nv)+2)
        if id+1 == len(vars(args)):
            L.append(l)
    printstr = niceprint(L)
    print(printstr)


def niceprint(L,mark="-"):
    printstr = []
    printstr.append("-"*len(L[0]))
    printstr.append(L[0])
    for id in range(1,len(L)):
        printstr.append("-"*max(len(L[id-1]),len(L[id])))
        printstr.append(L[id])
    printstr.append("-"*len(L[-1]))
    printstr = "\n".join(printstr)
    return printstr

from queue import Queue


class Vertex:
    def __init__(self, state):
        self.id = state
        self.connectedTo = {}
        self.InNode=set()
        self.flag=0
    def isNeighbor(self,nbr):
        if nbr in self.connectedTo:
            return True
        else:
            return False
    def addNeighbor(self, nbr, prob):
        self.connectedTo[nbr] = prob
    def addInNode(self,innode):
        self.InNode.add(innode)
    def delInode(self,innode):
        self.InNode.remove(innode)
    def getIndegree(self):
        return len(self.InNode)
    def __str__(self):
        return str(self.id) + ' connectedTo: ' + str([x.id for x in self.connectedTo])

    def getConnections(self):
        return self.connectedTo.keys()

    def getId(self):
        return self.id

    def getprob(self, nbr):
        return self.connectedTo[nbr]


class Graph:
    def __init__(self):
        self.vertList = {}
        self.numVertices = 0

    def addVertex(self, state):
        self.numVertices = self.numVertices + 1
        newVertex = Vertex(state)
        self.vertList[state] = newVertex
        return newVertex

    def getVertex(self, state):
        if state in self.vertList:
            return self.vertList[state]
        else:
            return None

    def __contains__(self, state):
        return state in self.vertList

    def addEdge(self, state_1, state_2, prob=0.0):
        if state_1 not in self.vertList:
            self.addVertex(state_1)
        if state_2 not in self.vertList:
            self.addVertex(state_2)
        if self.vertList[state_1].isNeighbor(self.vertList[state_2]):
            self.vertList[state_1].connectedTo[self.vertList[state_2]]+=prob
            #print("the same state")
        else:
            self.vertList[state_1].addNeighbor(self.vertList[state_2], prob)
        self.vertList[state_2].addInNode(self.vertList[state_1])
    def addEdge_2(self, state_1, state_2, prob):
        if state_1 not in self.vertList:
            self.addVertex(state_1)
        if state_2 not in self.vertList:
            self.addVertex(state_2)
        if self.vertList[state_1].isNeighbor(self.vertList[state_2]):
            current_pb=self.vertList[state_1].connectedTo[self.vertList[state_2]]
            up=current_pb[0]*prob[1]+prob[0]*current_pb[1]
            down=prob[1]*current_pb[1]
            self.vertList[state_1].connectedTo[self.vertList[state_2]]=[up,down]
            #print("the same state")
        else:
            self.vertList[state_1].addNeighbor(self.vertList[state_2], prob)

    def getVertices(self):
        return self.vertList.keys()

    def __iter__(self):
        return iter(self.vertList.values())

    def get_all_states(self):
        return self.vertList.keys()

    def hasCircle(self):
        VerticesList=self.vertList.values()
        q=Queue()
        for item in VerticesList:
            if item.flag==0 and item.getIndegree()==0:
                q.put(item)
        while not q.empty():
            cur=q.get()
            cur.flag=1
            Neighbors=cur.getConnections()
            print(Neighbors)
            for nbr in Neighbors:
                nbr.delInode(cur)
                if nbr.getIndegree()==0:
                    q.put(nbr)
        for itr in VerticesList:
            if itr.flag==0:
                return "Exsists Circle"
        return "No Circle"
    def Print(self):
        VerticesList = self.vertList.values()
        for item in VerticesList:
            if (item.getConnections()):
                #print(item.getConnections())
                for itr in item.getConnections():
                    print(item.getId(),itr.getId(),item.getprob(itr))
if __name__ == '__main__':
    G=Graph()
    G.addEdge("1", "2", 0.2)
    G.addEdge("1", "3", 0.3)
    G.addEdge("2", "4", 0.7)
    G.addEdge("2", "5", 0.2)
    G.addEdge("3", "6", 0.3)
    G.addEdge("3", "7", 0.4)
    G.addEdge("4", "2", 0.4)
    G.Print()
    #print(G.hasCircle())
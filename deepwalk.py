import numpy as np
import networkx as nx
from gensim.models import Word2Vec

class DeepWalk:
    def __init__(self, graph, walk_length=10, num_walks=100, dimensions=64, window_size=5, min_count=1, sg=1):
        self.graph = graph
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.dimensions = dimensions
        self.window_size = window_size
        self.min_count = min_count
        self.sg = sg

    def random_walk(self, start_node):
        walk = [start_node]
        for _ in range(self.walk_length):
            cur = walk[-1]
            neighbors = list(self.graph.neighbors(cur))
            if neighbors:
                walk.append(np.random.choice(neighbors))
            else:
                break
        return walk

    def generate_walks(self):
        walks = []
        nodes = list(self.graph.nodes())
        for _ in range(self.num_walks):
            np.random.shuffle(nodes)
            for node in nodes:
                walk = self.random_walk(node)
                walks.append(walk)
        return walks

    def learn_embeddings(self, walks):
        # Convert walks to strings
        walks = [[str(node) for node in walk] for walk in walks]
        model = Word2Vec(walks, vector_size=self.dimensions, window=self.window_size, min_count=self.min_count, sg=self.sg)
        return model.wv

def main():
    # Load the Karate Club graph
    G = nx.karate_club_graph()
    
    # Create a DeepWalk instance
    deepwalk = DeepWalk(graph=G, walk_length=10, num_walks=100, dimensions=64)
    
    # Generate random walks
    walks = deepwalk.generate_walks()
    
    # Learn embeddings
    embeddings = deepwalk.learn_embeddings(walks)
    
    # Output the learned embeddings
    for node in G.nodes():
        print(f'Node: {node}, Embedding: {embeddings[str(node)]}')

if __name__ == "__main__":
    main()

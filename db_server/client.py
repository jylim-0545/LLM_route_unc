import grpc
import faiss_pb2
import faiss_pb2_grpc

def run():
    channel = grpc.insecure_channel('localhost:5050')
    stub = faiss_pb2_grpc.FaissStub(channel)
    
    while (query := input("Type Question:")) != "exit":
    
        response = stub.Search(faiss_pb2.SearchRequest(query = query, top_k = 3))
        
        for context, score in zip(response.contexts, response.scores):
            print(f"Context: {context} | Distance: {score} ")
            print(100*"-")    

if __name__ == '__main__':
    run()

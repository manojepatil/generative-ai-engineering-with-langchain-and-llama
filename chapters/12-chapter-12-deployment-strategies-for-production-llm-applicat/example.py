# deployment_strategies.py

import time
import threading

class LLMService:
    """
    Simulated LLM service representing a LangChain/LlamaIndex application.
    It processes queries and simulates varying response times.
    """
    def __init__(self, name, processing_time_factor=1.0):
        self.name = name
        self.processing_time_factor = processing_time_factor
        self.request_count = 0
        print(f"LLM Service '{self.name}' initialized with processing factor: {self.processing_time_factor}")

    def process_query(self, query):
        """Simulates processing an LLM query."""
        self.request_count += 1
        processing_time = len(query) * 0.01 * self.processing_time_factor  # Simulate variable load
        time.sleep(processing_time)
        response = f"Service {self.name} processed '{query}' in {processing_time:.2f}s."
        print(response)
        return response

class LoadBalancer:
    """
    A simple load balancer to distribute requests among multiple LLM services.
    Represents a strategy for scalability and reliability.
    """
    def __init__(self, services):
        self.services = services
        self.next_service_index = 0
        print(f"Load Balancer initialized with {len(services)} services.")

    def distribute_query(self, query):
        """Distributes a query to the next available service in round-robin fashion."""
        selected_service = self.services[self.next_service_index]
        self.next_service_index = (self.next_service_index + 1) % len(self.services)

        print(f"Routing query '{query}' to service '{selected_service.name}'...")
        return selected_service.process_query(query)

def client_request(load_balancer, query_id):
    """Simulates a client making a request to the load balancer."""
    query = f"Query {query_id}"
    start_time = time.time()
    load_balancer.distribute_query(query)
    end_time = time.time()
    print(f"Client received response for '{query}' in {end_time - start_time:.2f}s.")

if __name__ == "__main__":
    # 1. Initialize multiple LLM services (representing instances for scalability)
    # Different processing factors can simulate heterogeneous instances or varying load.
    service_a = LLMService("Service-A", processing_time_factor=1.0)
    service_b = LLMService("Service-B", processing_time_factor=0.8) # Faster instance
    service_c = LLMService("Service-C", processing_time_factor=1.2) # Slower instance

    llm_services = [service_a, service_b, service_c]

    # 2. Deploy a Load Balancer to distribute requests
    # This is a common pattern for managing scalability and high availability.
    production_load_balancer = LoadBalancer(llm_services)

    # 3. Simulate concurrent client requests
    # This demonstrates how the system handles multiple users.
    print("\nSimulating client requests:")
    queries_to_send = 10
    client_threads = []

    for i in range(queries_to_send):
        # Create a new thread for each client request to simulate concurrency
        thread = threading.Thread(target=client_request, args=(production_load_balancer, i + 1))
        client_threads.append(thread)
        thread.start()
        time.sleep(0.1)  # Stagger requests slightly

    # Wait for all client threads to complete
    for thread in client_threads:
        thread.join()

    print("\nDeployment simulation complete.")
    for service in llm_services:
        print(f"Service '{service.name}' handled {service.request_count} requests.")

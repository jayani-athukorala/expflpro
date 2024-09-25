import flwr as fl

# Start Flower server with a simple configuration
def main():
    fl.server.start_server(config=fl.server.ServerConfig(num_rounds=1))

if __name__ == "__main__":
    main()

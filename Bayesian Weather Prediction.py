from pomegranate import *

# Define the variables and their states
outlook = DiscreteDistribution({'sunny': 0.4, 'rainy': 0.3, 'cloudy': 0.3})
temperature = DiscreteDistribution({'hot': 0.3, 'mild': 0.4, 'cool': 0.3})
humidity = DiscreteDistribution({'high': 0.6, 'normal': 0.4})
forecast = ConditionalProbabilityTable(
    [
        ['sunny', 'high', 'good', 0.8],
        ['sunny', 'high', 'bad', 0.2],
        ['sunny', 'normal', 'good', 0.6],
        ['sunny', 'normal', 'bad', 0.4],
        ['rainy', 'high', 'good', 0.3],
        ['rainy', 'high', 'bad', 0.7],
        ['rainy', 'normal', 'good', 0.2],
        ['rainy', 'normal', 'bad', 0.8],
        ['cloudy', 'high', 'good', 0.5],
        ['cloudy', 'high', 'bad', 0.5],
        ['cloudy', 'normal', 'good', 0.4],
        ['cloudy', 'normal', 'bad', 0.6],
    ],
    [outlook, humidity]
)

# Define the nodes
outlook_node = State(outlook, name='outlook')
temperature_node = State(temperature, name='temperature')
humidity_node = State(humidity, name='humidity')
forecast_node = State(forecast, name='forecast')

# Create the Bayesian network
network = BayesianNetwork('Weather Forecasting')
network.add_nodes(outlook_node, temperature_node, humidity_node, forecast_node)

# Add edges between nodes
network.add_edge(outlook_node, forecast_node)
network.add_edge(temperature_node, forecast_node)
network.add_edge(humidity_node, forecast_node)

# Finalize the structure
network.bake()

# Perform inference on the network
observations = {'outlook': 'sunny', 'humidity': 'high'}
beliefs = network.predict_proba(observations)
for node, belief in zip(network.states, beliefs):
    print(node.name, belief.parameters[0])

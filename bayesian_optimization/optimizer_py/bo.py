import sys
sys.path.append('Release')
import simulator

# Call dummy function to test binding
result = simulator.dummy_simulation(3.0)
print("Result:", result)
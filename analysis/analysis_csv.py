import pandas as pd
import matplotlib.pyplot as plt

'''
    The Following CSV file was generated using this NgQL Query :

        MATCH (n)-[r]->()
    WITH n, COUNT(r) AS outdegree

    MATCH (n)<-[r]-()
    WITH n, COUNT(r) + outdegree AS total_degree

    RETURN n.entity.name AS node, total_degree
    ORDER BY total_degree ASC
'''
# Load the CSV file
df = pd.read_csv('/Users/kunjrathod/MedRAG/scripts/NebulaGraph Studio result.csv')  # Adjust the path to your file

# Plot the distribution of total degrees
plt.hist(df['total_degree'], bins=50, edgecolor='black')
plt.xlabel('Total Degree')
plt.ylabel('Frequency')
plt.title('Distribution of Total Degrees')
plt.show()

# Print basic statistics
print(df['total_degree'].describe())

'''
    BIOKG:

    count    57608.000000
    mean        88.619567
    std        226.813066
    min          2.000000
    25%          7.000000
    50%         36.000000
    75%         81.000000
    max      13939.000000
    
'''
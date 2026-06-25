import pandas as pd
import matplotlib
matplotlib.use('Agg') # Headless mode for cluster
import matplotlib.pyplot as plt
import os

# 1. Load data
if not os.path.exists('scale_results.csv'):
    print("Error: scale_results.csv not found. Run scale_test.sh first!")
    exit()

df = pd.read_csv('scale_results.csv')

plt.figure(figsize=(12, 7))

# 2. Plotting the two methodologies
for label in ['Hybrid_Switching', 'Always_Lossy']:
    subset = df[df['Method'] == label]
    plt.plot(subset['Size'], subset['Time_us'], marker='o', label=label, linewidth=2.5)

# 3. Identify and Annotate the Switching Point
# The threshold in your libhybrid.cpp was 20480
threshold = 20480
plt.axvline(x=threshold, color='red', linestyle='--', alpha=0.6, label='Switching Threshold')

# 4. Highlight the "Optimization Gain"
# This is the gap at the small message sizes
small_messages = df[df['Size'] < threshold]
if not small_messages.empty:
    plt.annotate('Optimization Gain\n(Middleware vs Base Paper)', 
                 xy=(1024, small_messages[small_messages['Method']=='Always_Lossy']['Time_us'].iloc[0]), 
                 xytext=(5000, small_messages[small_messages['Method']=='Always_Lossy']['Time_us'].iloc[0] * 1.5),
                 arrowprops=dict(facecolor='black', shrink=0.05),
                 fontsize=10, fontweight='bold')

# 5. Formatting
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Message Size (Number of Floats)')
plt.ylabel('Total Execution Time (us)')
plt.title('HPC Analysis: Hybrid Switching vs. Static Lossy Compression')
plt.grid(True, which="both", ls="-", alpha=0.3)
plt.legend()

# 6. Save the figure
plt.savefig('final_comparison_plot.png', dpi=300)
print("Graph saved as final_comparison_plot.png")
import pandas as pd
import plotly.express as px
import math

# Load the tracking data from the CSV file into a DataFrame
df = pd.read_csv('trajectories.csv')
#clean data here?

df_sorted = df.sort_values(by='trackID')
counts = df['trackID'].value_counts()

# Identify tracking_ids with fewer than x amt of occurrences
ids_to_keep = counts[counts >= 25].index

# Filter the original DataFrame based on ids_to_keep
df_filtered = df_sorted[df_sorted['trackID'].isin(ids_to_keep)]

# Display the filtered DataFrame
df_filtered = df_filtered.sort_values(by='frame')
#df_filtered.to_csv('testData.csv', index=False)
#print("testData.csv saved")


# Plot the trajectories with Plotly Express add symbol = trackID if want to sort by class color
fig = px.line(df_filtered, x='xCenter', y='yCenter', color='Class', symbol = 'trackID', title='Object Trajectories')

# Reverse the y-axis to match image coordinates
fig.update_yaxes(autorange="reversed")

# Mark the starting and ending points
# This is an additional step to highlight start and end points

# Group by trackID and get the first and last points
start_points = df_filtered.groupby('trackID').first().reset_index()
end_points = df_filtered.groupby('trackID').last().reset_index()

# Add the start and end points to the plot
fig.add_scatter(x=start_points['xCenter'], y=start_points['yCenter'],
                mode='markers', name='Start', marker=dict(color='green', size=10, symbol='circle'))

fig.add_scatter(x=end_points['xCenter'], y=end_points['yCenter'],
                mode='markers', name='End', marker=dict(color='red', size=10, symbol='x'))

# Update the layout to customize axis titles and figure size
fig.update_layout(
    xaxis_title="X Coordinate",
    yaxis_title="Y Coordinate",
    width=1000,
    height=800
)

# Show the plot
fig.show()

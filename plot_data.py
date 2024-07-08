import pandas as pd
import plotly.express as px

# Load the tracking data from the CSV file into a DataFrame
df = pd.read_csv('trajectories.csv')

# Plot the trajectories with Plotly Express
fig = px.line(df, x='xCenter', y='yCenter', color='trackID', title='Object Trajectories')

# Reverse the y-axis to match image coordinates
fig.update_yaxes(autorange="reversed")

# Mark the starting and ending points
# This is an additional step to highlight start and end points

# Group by trackID and get the first and last points
start_points = df.groupby('trackID').first().reset_index()
end_points = df.groupby('trackID').last().reset_index()

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

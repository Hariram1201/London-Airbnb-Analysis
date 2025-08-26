library(ggplot2)  # Collection of packages to create static, interactive and customisable plots
library(plotly) #Collection of packages used to create interactive visualisations 

boxPlot <- function(dataset, xVar, title, xTitle){
  
  "
  Creates and displays a box and whisker plot with jittered points for visualizing the distribution of a variable.
  
  Parameters:
  -----------
  dataset : data.frame
      The dataset containing the data to be plotted.
  xVar : numeric vector
      The variable (numeric vector) to plot in the boxplot.
  title : character
      The title of the plot.
  xTitle : character
      The label for the x-axis.
  
  Returns:
  --------
  NULL
      The function prints the plot and returns nothing.
  "
  
  plot <- ggplot(dataset, aes(x = xVar, y = "")) +
    geom_boxplot(fill = "lightblue", color = "black", width = 0.25,
                 outlier.size = 1, outlier.shape = NA) +
    geom_jitter(color = "black", width = 0.2, alpha = 0.5) +
    labs(title = title, x = xTitle) +
    theme(
      plot.title = element_text(hjust = 0.5, size = 18, face = "bold"),
      axis.title = element_text(size = 14),
      axis.text = element_text(size = 12)
    ) +
    theme_minimal()
  
  print(plot)
}

histogram <- function(dataset, xVar, title, xTitle, yTitle){
  
  "
  Visualizes the data by displaying a histogram of the specified variable against frequency.
  
  Parameters:
  -----------
  dataset : data.frame
      The dataset containing the data to plot.
  xVar : string
      The name of the variable/column to plot on the x-axis.
  title : string
      The title of the histogram plot.
  xTitle : string
      The label for the x-axis.
  yTitle : string
      The label for the y-axis.
  
  Returns:
  --------
  None
      Displays a histogram plot with counts annotated on the bars.
  "
  
  plot <- ggplot(dataset, aes_string(x = xVar)) +
    geom_histogram(bins = 30, fill = 'blue', color = 'black') +
    geom_text(stat = "bin", aes(label = ..count.., y = ..count..), vjust = -0.5) +
    labs(title = title, x = xTitle, y = yTitle) +
    theme(
      plot.title = element_text(hjust = 0.5, size = 16, face = "bold"),
      axis.title = element_text(size = 14),
      axis.text = element_text(size = 12),
      panel.background = element_rect(fill = "lightgray"),
      panel.grid.major = element_line(color = "white", size = 0.5)
    ) + 
    theme_minimal()
  
  print(plot)
}

violin <- function(dataset, xVar, yVar, title, xTitle, yTitle){
  
  "
  Creates and displays a violin plot to show the distribution of a numeric variable across categories.
  
  Parameters:
  -----------
  dataset : data.frame  
      The dataset containing the data to be visualized.
  xVar : character  
      The name of the categorical variable to plot on the x-axis.
  yVar : character  
      The name of the numeric variable to plot on the y-axis.
  title : character  
      The title of the plot.
  xTitle : character  
      The label for the x-axis.
  yTitle : character  
      The label for the y-axis.
  
  Returns:
  --------
  NULL  
      Displays a violin plot. The function returns nothing.
  "
  
  plot <- ggplot(dataset, aes(x = .data[[xVar]], y = .data[[yVar]])) +
    geom_violin(fill = "lightgreen", color = "black", alpha = 0.7) +
    labs(title = title, x = xTitle, y = yTitle) +
    theme(
      plot.title = element_text(hjust = 0.5, size = 16, face = "bold"),
      axis.title = element_text(size = 14),
      axis.text = element_text(size = 12),
      panel.background = element_rect(fill = "lightgray"),
      panel.grid.major = element_line(color = "white", size = 0.5)
    ) + 
    theme_minimal()
  
  print(plot)
}

facet <- function(dataset, xVar, yVar, colourVar, title, xTitle, yTitle){
  
  "
  Creates and displays a faceted scatter plot with linear trend lines, grouped by a categorical variable.
  
  Parameters:
  -----------
  dataset : data.frame  
      The dataset containing the variables to be plotted.
  xVar : character  
      The name of the numeric variable for the x-axis.
  yVar : character  
      The name of the numeric variable for the y-axis.
  colourVar : character  
      The name of the categorical variable used to facet and colour the plots.
  title : character  
      The title of the plot.
  xTitle : character  
      The label for the x-axis.
  yTitle : character  
      The label for the y-axis.
  
  Returns:
  --------
  NULL  
      Displays a faceted scatter plot with trend lines. The function returns nothing.
  "
  
  plot <- ggplot(dataset, aes(x = .data[[xVar]], y = .data[[yVar]], color = .data[[colourVar]])) +
    geom_point(size = 0.5, alpha = 0.6) +
    geom_smooth(method = "lm", se = FALSE) +
    facet_wrap(vars(.data[[colourVar]])) + 
    labs(title = title, x = xTitle, y = yTitle) +
    theme_minimal() +
    theme(legend.position = "none", plot.title = element_text(hjust = 0.5))  
  
  print(plot)
}

linear <- function(dataset, xVar, yVar, title, xTitle, yTitle){
  "
  Creates a scatter plot with a linear regression line to show the relationship between two variables.

  Parameters:
  -----------
  dataset : data.frame
      The dataset containing the variables to be plotted.
  xVar : string
      The name of the variable to be used on the x-axis.
  yVar : string
      The name of the variable to be used on the y-axis.
  title : string
      The title of the plot.
  xTitle : string
      The label for the x-axis.
  yTitle : string
      The label for the y-axis.

  Returns:
  --------
  None
      Displays a scatter plot with a linear regression line and confidence interval.
  "
  
  # Create the scatter plot with a linear regression fit line
  plot <- ggplot(dataset, aes(x = .data[[xVar]], y = .data[[yVar]])) +
    geom_point(color = "red", size = 1) +  # Red data points
    geom_smooth(method = "lm",            # Linear regression line
                color = "blue",           # Blue line
                se = TRUE,                # Show shaded confidence interval
                fill = "lightblue") +     # Shading color
    labs(title = title, x = xTitle, y = yTitle) +  # Add titles and labels
    theme_minimal() +                               # Use minimal theme
    theme(plot.title = element_text(hjust = 0.5))   # Center the plot title
  
  print(plot)
}

twoLinear <- function(dataset, xVar, yVar, colourVar, title, xTitle, yTitle, colourTitle){
  "
  Creates a scatter plot with separate linear regression lines for different groups within a categorical variable.

  Parameters:
  -----------
  dataset : data.frame
      The dataset containing the variables to be plotted.
  xVar : string
      The name of the variable to be plotted on the x-axis.
  yVar : string
      The name of the variable to be plotted on the y-axis.
  colourVar : string
      The name of the categorical variable used to color and group the data.
  title : string
      The title of the plot.
  xTitle : string
      The label for the x-axis.
  yTitle : string
      The label for the y-axis.
  colourTitle : string
      The label for the legend corresponding to the color variable.

  Returns:
  --------
  None
      Displays a grouped scatter plot with individual linear regression lines for each group.
  "
  
  # Generate the grouped scatter plot with linear regression lines by category
  plot <- ggplot(dataset, aes(x = .data[[xVar]], y = .data[[yVar]], color = .data[[colourVar]])) +
    geom_point(alpha = 0.6, size = 1) +                        # Semi-transparent points
    geom_smooth(method = "lm", se = FALSE, size = 1.2) +       # Line of best fit per group, no CI shading
    labs(title = title, x = xTitle, y = yTitle, color = colourTitle) +  # Axis and legend titles
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5))              # Center the plot title
  
  print(plot)
}

threeDimLin <- function(dataset, xVar, yVar, zVar, title, xTitle, yTitle, zTitle){
  "
  Creates a 3D scatter plot to visualize the relationship between three numerical variables.

  Parameters:
  -----------
  dataset : data.frame
      The dataset containing the variables to be plotted.
  xVar : string
      The name of the variable to be plotted on the x-axis.
  yVar : string
      The name of the variable to be plotted on the y-axis.
  zVar : string
      The name of the variable to be plotted on the z-axis and used for coloring.
  title : string
      The title of the 3D plot.
  xTitle : string
      The label for the x-axis.
  yTitle : string
      The label for the y-axis.
  zTitle : string
      The label for the z-axis and the color legend.

  Returns:
  --------
  None
      Displays a 3D scatter plot using plotly.
  "
  
  # Generate 3D scatter plot using Plotly with coloring based on the z variable
  plot <- plot_ly(dataset,
                  x = ~.data[[xVar]],
                  y = ~.data[[yVar]],
                  z = ~.data[[zVar]],
                  type = "scatter3d",
                  mode = "markers",
                  color = ~.data[[zVar]], colors = "Blues",
                  marker = list(size = 4)) %>%
    layout(title = title,
           scene = list(
             xaxis = list(title = xTitle),
             yaxis = list(title = yTitle),
             zaxis = list(title = zTitle)
           ))
  
  print(plot)
}

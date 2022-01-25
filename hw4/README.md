# Data Skills 2 Homework 4

### Due Sunday November 7th before midnight

For your final assignment you will be creating an interactive choropleth.  To begin, you will choose:

  * Any one **continent**
  * Any one **country within that continent**
  * Two variables that you will use to scale colors on the choropleth:
    - For the continet choropleth, that variable will be at the country level
	- For the country choropleth, that variable will be at any reasonable subdivisions within that country (e.g. states, provinces)
	
You will create a single Bokeh plot which includes two interactive elements:

  * One element to select the country, which defaults to the whole continent
  * One element to select the variable to display on the map
  
**Your code must be generalized using functions, so that you could easily add more countries within that continent to the geographic options, and more variables to the data options.**  If any of your code from homework 3 can be leveraged to accomplish this, I highly encourage you to do so.  Furthermore, if any of your functions from this assignment can be written so that they also apply to your final project, even better.

Your final repo should include:

  * Your Jupyter notebook that contains all your code and your interactive plot
    - Include a markdown cell at the end that includes around 5 lines of discussion, particularly regarding your attempts to leverage generalized code from HW3 and/or apply generalized code to your final project
  * Any shapefiles you downloaded
  * Any data you retrieved manually
    - You are free to use Pandas DataReader, or other web retrieval tools, but you are not required to do so.  If any of your data was retrieved in zip format, please be sure to commit the unizpped versions for ease of grading.
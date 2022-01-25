# Data Skills 2: Homework 3
### Data Visualization

__Due Date: Sunday, October 31st before midnight__

Your assignment for this week will combine all of the data visiualization skills we've practiced for the last few weeks, with some data developed by the Urban Institute, which will be discussed in class.

__The Data:__

The Urban Institute hosts an open data portal on education in the United States.  Researchers at Urban clean and assemble a wide variety of data, and then make the organized data available for download.  In this case, you will retrieve college data from the Education Data Portal:

https://educationdata.urban.org/data-explorer/colleges/

You must use their interface, which builds a SQL query behind-the-sceens as you select options, then returns custom-built csv documents to you for download.  Certain data is only available for certain years; in this case we will be working with the date ranges 2001-2016, because that range enables a large number of options.

Depending on the options you select, you may receive more than one data file that needs to be merged together.  You will also receive a data dictionary.

_You must commit the downloaded csv files with your code into your repository._ There are too many ways to customize your query, and I am not giving any guidelines on what options you select beyond the date range listed above, so the graders will not be able to reproduce your data.

__The Task:__

You work in the education policy field, and you are proposing your organization use the Urban Institute data to study higher education.  Their nicely organized data offers a lot of opportunities to gain insights into the field.

The result of this project will be:
  1. One Jupyter notebook with all your code, plots, and writeup.
     - We use Jupyter notebooks here because of the interactive widgits
	 - Before committing your final notebook, click the "Kernal" menu and select "Restart & Run All", then save it and commit.  This will help make sure your code works in sequence, and will make it easier for the TAs to grade.
     - Note one key hazard of working in Jupyter notebooks: Jupyter cells do not count as a method of organization or substitute for functions!
  2. All data loading and merging code (but not downloading, which must use the Urban web interface)
  3. The creation of plots:
     - _Two_ static plot from MatPlotLib/Pandas/Seaborn _(OR one plot with two related subplots)_.  These plots should be a "headline" result that you would, for example, embed in a writeup to your colleagues about why they should consider using this data.  Note that you will not be creating such a writeup - we're just framing the goal of our data visualizations. You should show effort in your code toward making it a nice plot.
     - _Two_ interactive plots from Bokeh _(OR one plot with multiple panels, OR two related plots on one grid, etc)._  This plot should allow you and your colleagues to explore an assortment of interesting questions you could answer with this data.

You will then include an approximately 5-10 line writeup in a Jupyter cell (change the type from Code to Markdown) about how the assignment went and what you observed using data visualization.  In your writeup, focus particularly on generalization - were you able to use functions with your code in a way that lets you make changes easier later, or that makes your code more legible for someone else to read?  How was the balance between using MatPlotLib and Seaborn?  Was there something you discovered by visualizing the data that you didn't expect, or couldn't see any other way?  How did you inform your plot selection - did you have a specific research question, or were you just exploring?

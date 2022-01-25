# Data_Skills_2_assignment_2

### Due Wednesday October 20th before midnight

Your task today is to parse the US Energy Information Administration's annual energy outlook reports.  The goal is code that summarizes the predictions in a table for a given year, so it can be re-run on every year with a similar format.  You will just be working with 2019 and 2018.  

Fortunately the required information is availble in the "key takeaway" sections at the beginning of the documents, so you can ignore the rest (e.g. pages 9-28 in the 2019 table of contents).  The end result should be two (one for each year) csv documents with the columns "energy type", "price", "emissions", "production", "export/import".  There should be five rows of data, one for each energy type of "coal", "nuclear", "wind", "solar", and "oil".  The values in the other columns should be a rough categorization of the predictions - will price go up or down, will emissions increase or decrease, will exports increase or decrease, and so on.  The exact details are open to what you think is best, as long as it helps summarize the key takeaways.  Obviously if the document doesn't say anything about a certain value in the table, it should be a missing value or a zero.  

  - Remember to break your work into bite-size pieces, and then generalize!
      - First, focus on one year.
      - Second, focus on one value.
      - Third, focus on one subset of the text that contains a relevant outcome
          - When you have something working for one subset, generalize it to the whole document 
	  - When you have something working for one value, generalize it to the other values.
	  - When you have something working for one whole year, generalize it to the other year.
  - You should retrieve the pdf documents from their respective urls within your code.
  - As we did in the NLP lecture, test your code on small blocks of text rather than the entire document.
  - Remember all we've talked about for good testing and debugging practices.
  - Ask for help; speak with the TAs, come see or email me, or use Ed Discussion
  - The final format of the two tables is not rigid.  If you think a slight change (e.g. to a column description, or adding a new column or row) would make the data easier to summarize, you can do that.
  - Please include a small comment block (around 5-10 lines, give or take) that outlines the logic of your code, and points out any remaining weaknesses.
  - Your code must utilize functions appropriately.
  - Do as much as you can - partial credit counts!

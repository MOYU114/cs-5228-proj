# Project

Other pages: [ [Overview](https://canvas.nus.edu.sg/courses/93866/pages/overview) ] [ [Errata](https://canvas.nus.edu.sg/courses/93866/pages/errata) ] [ [Project](https://canvas.nus.edu.sg/courses/93866/pages/project) ] [ [Midterm & Quiz](https://canvas.nus.edu.sg/courses/93866/pages/midterm-and-quiz) ] [ [Notebooks](https://canvas.nus.edu.sg/courses/93866/pages/lecture-notebooks) ] [ [Benchmarks](https://canvas.nus.edu.sg/courses/93866/pages/performance-benchmarks) ] [ [Policies](https://canvas.nus.edu.sg/courses/93866/pages/policies) ]

*Last update: 4 Aug, 2026*

 

## Overview

The purpose of the final project is for you to show how you perform data mining in a practical setting. Given a dataset and a defined task, you need to select appropriate techniques to solve this task, justify design and implementation issues, as well as interpret your results and assess any limitations of your approach. We tried to design the project to make the task both interesting and relevant. As often emphasized in the lectures, there is rarely one single best way to solve a data mining task, and many steps will therefore benefit or even require your own creativity to come up with appropriate solutions. While this project will require a certain amount of effort, we also hope that you will have fun completing it, and that it will be a valuable learning experience.

In this project, we look into the rental market in Singapore, more specifically the rental market for HDB flats. Accommodation is one of the biggest financial burdens in Singapore. Particularly in the most recent years, both the resale and rental market saw a significant rise in prices and rental rates. The figure below shows an example listing for an HDB flat on [RentInSingapore.com.sgLinks to an external site.](https://rentinsingapore.com.sg/).

![Screenshot - RentInSingapore Listing](https://canvas.nus.edu.sg/courses/93866/files/9565692/preview)

For non-homeowners (e.g., singles, young couples waiting for a BTO flat, or foreign workers with or without family) who do not have the financial means to go with the resale market – or are not eligible – renting is often the only practical solution (even if only temporarily). Thus, potential renters therefore want to know what they can get for their money, where they can best save money, and simply spot bargains and/or rip-offs. On the other hand, landlords and real estate agents aiming to maximize rental rates want to know how to best present and advertise their flats. Also, with the limited amount of available land, affordable housing is a major issue of the Singaporean government, which may deploy "cooling measures" to influence the housing market. In short, there are many stakeholders that rely and benefit from a deeper understanding of the Singaporean housing market.

More specifically, the main goals of this project is the prediction of rental rates of HDB flats, given the information about an HDB flat (e.g., size, \#rooms, location), your task is to predict the expected rent based on historical data. This regression task is implemented as a Kaggle InClass Competition. Note that the focus is less on your final results but on the understanding of the data, task and results, as well as on the motivation and justification of your approach, incl. discussions about potential alternatives and limitations.

 

## Dataset

### Core Dataset

The core dataset of rental rates for HDB flats has been collected and consolidated from DATA.GOV.SG [[1Links to an external site.](https://data.gov.sg/datasets/d_c9f57187485a850908655db0e8cfe651/view),[2Links to an external site.](https://data.gov.sg/datasets/d_17f5382f26140b1fdae0ba2ef6239d2f/view)]. In Singapore, HDB owners need to \href{https://www.hdb.gov.sg/business/estate-agents-and-salespersons/renting-out-a-flat-or-bedroom/regulations-for-renting-out-flats}{apply for approval} to rent our their flat or single rooms. This information about approved applications is made available on data.gov.sg; of course without any personal information. Each entry in this dataset comes with the

- Rent approval date (e.g., 2022-05, 2023-10)
- Town (e.g., Punggol, Bedok, Jurong West)
- Block (e.g., 650, 321A, 95B)
- Street name (e.g., Punggol Walk, Boon Lay Ave)
- Monthly rent in SGD

The dataset contains all the approved applications from Jan 2021 to Jun 2026.

Using additional data sources, we have extended each data entry by several additional attributes such as the flat type and size, as well as the lease commence date of the block. Overall, this core dataset consists of 9 attributes. While the meanings of all attributes should be rather self-explanatory, if you still have questions, you can ask your question on the Canvas or via email.

 

### Auxiliary Dataset

While a flat's most basic features (e.g., size and age) certainly affect the rent, other factors are also very important when it comes to assessing the rental rate for a given flat. We therefore provide some additional data to enrich the core dataset described above:

- Apart from the information about the individual apartments, we also provide some additional information about HDB block, including its number of floors, its year of completion, its geolocation as latitude-longitude pair, and other details 
- Location is often touted to be one of the most important aspects of accommodation. Apart from exact coordinates of HDB blocks, we also collected the location of important "landmarks" such as MRT stations (both existing and planned stations), shopping malls, and primary schools.
- Rental rates are not decided on in isolation but also depend on the overall "climate" like the general economic situation. As some indicators reflecting the economy, we collected the stock prices for the largest [Singapore-based companies by market capitalizationLinks to an external site.](https://companiesmarketcap.com/singapore/largest-companies-in-singapore-by-market-cap/), as well as the {[Certificate of Entitlement (COE)Links to an external site.](https://en.wikipedia.org/wiki/Certificate_of_Entitlement) prices for the same time interval as the core dataset.

 

**The complete dataset and more details will be available on Kaggle once the project is officially launched.**

Some additional comments

- You are not required to use all the attributes provided. Some features might be more or less useful. Particularly at the beginning you might want to first focus on a subset of more "convenient" attributes to get a good sense of the data and the task.
- While not mandatory, you are welcome to collect additional data if you think it might help the task.
- EDA, data cleaning/preparation, model selection, hyperparameter tuning, etc. gives you a lot of flexibility. In fact, it is unlikely that you have enough time to try out all possibilities. Do not worry, this is on purpose! It is up to you to decide which data and attributes you deem useful for addressing the different required steps.

 

## Deliverables

### Progress Report

The progress report will be a simple slide deck as a PDF document of approx. 10-15 slides. The purpose of the progress report is two-fold: (a) to give us a chance to check if your project goes into the right direction, and (b) to provide you with a little incentive to start early. There is no official layout or structure. As the name suggests, it should outline your progress with your project work (e.g., goals and questions, understanding of task and the data, first design decisions and/or results, but also issues/challenges/obstacles that you are facing). The last 1-2 slides should outline the next steps until the end of the project.

- Deadline: **Sun, Oct 4** (not a hard deadline; earlier submissions are encouraged and welcome)

**Note:** You are welcome to submit your progress report earlier. Ideally, this will in turn give you earlier feedback, but also allow us to better balance the workload. The progress report will not be explicitly graded but not submitting any report will negatively affect your final grade.



### Final Report

The final report will be a PDF document in the format of a scientific paper of at most **8 pages** including tables, plots and figures, but excluding references and the appendix. The appendix may contain supplementary content but should be used sparingly. As a rule of thumb, the report should be readable and completely comprehensible without the appendix. The appendix typically may include plots or tables that elaborate on the results of your EDA or your evaluation. For the layout and presentation of the report, we provide [templates](https://canvas.nus.edu.sg/courses/93866/files/9565679?wrap=1)[Download templates](https://canvas.nus.edu.sg/courses/93866/files/9565679/download?download_frd=1) for Word and LaTeX.

#### Structure & Content

Your report should include the name and student IDs of all team members as well as your team name – pick something cool :). Please also include a breakdown of your workload, i.e., some overview what team member was (mainly) responsible for each part of the project. This can be a table, Gantt chart, etc. to be added to the appendix. While the overall structure of the report is up to you, it should cover the following aspects – although this might differ from your exact project task:

- **Motivation.** Motivate and outline the goals and questions you address. Why is this work important, what are the challenges, will benefits from the results. In short, what problem are you trying to solve? For example, simply aiming for a top rank on Kaggle is not a sufficient motivation :).
- **Exploratory Data Analysis & Preprocessing.** Explain and justify your approach to understand the data, and how it informed your data preprocessing steps (e.g., data reduction, data transformation, outlier removal, feature generation).
- **Data Mining Methods.** Describe how you chose and applied appropriate data mining techniques. This description should include which techniques you used, how you chose their hyperparameters, etc. Note that you do not need to explain the techniques themselves. However, in case of more advanced methods or models, you should add relevant references.
- **Evaluation & Interpretation.** Evaluate and compare the performance of different methods. Discuss which method(s) performed best and why. Understand in what cases your methods perform bad, and discuss principle limitations and potential future steps for improvement.

 

#### Submission

The final submission contains both the report as a PDF document as well as your source code, uploaded to Canvas in a zipped folder. Instead of the source code, you can also add a link to a GitHub repository. Note that the reproducibility of your approach is part of the grading (see below) which includes the organization, documentation, and readability of your code.

- Deadline: Xxx**, Nov XX, 11.59 pm
  **

 

## Grading

### Marking Guide

The individual projects are likely to different greatly. To keep it fair, all projects will be based on the same fundamental criteria. In a nutshell, a good grade requires that your approach is methodologically sound, and that the outcome – mainly the report but also your source code – is of a high quality. In more detail, we weigh the core criteria for the grading as follows

**Methodological Quality (60%).** While the exact distribution may depend on your chosen project and approach, methodological quality generally covers the following aspects:

- Data preparation: appropriate preprocessing methods are chosen (informed by the results of the EDA) and correctly implemented; missing values, categorical attributes, etc. are handled correctly.
- Visualization: appropriate plots, figures and tables are used to visualize results, architectures and work flows.
- Methods: applied methods are well motivated and correctly implemented; alternatives are discussed and design decision are justified.
- Evaluation: depending the proposed approach, the correct experiment setup and evaluation metrics are considered; common errors and principle limitations are evaluated and discussed.

**Quality of Report (30%).** The report describes your methodology and explains your results in a clear, concise and comprehensible manner. Related work should be appropriately referenced; the limit of 8 pages should not be exceeded (excluding references and appendix).

**Reproducibility (10%).** The code you submit is complete, well-organized ad readable. Simply put, it should be easy for an outsider to use and understand your code to retrace your steps and reproduce results.

 

### Team Assessment

By default, all members of a team will receive the same marks. However, we reserve ourselves the possibility to deviate from this approach in case different marks for different members are indeed warranted. To quantify this, but also to incentivize each team member to equally contribute, we will be using the the peer review system [TEAMMATESLinks to an external site.](https://teammatesv4.appspot.com/). For a smooth collaboration we recommend:

- Start the project early (at least to get a sense of the expected workload)
- Meet and discuss regularly (organize at least 1 or 2 weekly sessions where the whole team meets, if possible)
- Include some buffer time for any exams or unexpected and unplanned idle times.
- Have fun and make it a good experience!

If your team faces some internal conflicts, best let the teaching team know as soon as possible.

 
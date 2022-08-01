.. _purpose:


Purpose and contents
====================

This package generates plots of data with the following properties:

- One variable is quantitative.
- All other variables of interest, if any, are categorical.

The first two letters of the package name are meant to indicate one (Roman number I) quantitative (Q) variable. 

The subclass of data sets that contain a single quantitative variable (and possibly several categorical variables) abound in the biological sciences, which was the primary motivation for making this package.

There are six types of plots that iqplot can generate.

- **Plots with a categorical axis**

    + `Box plots <https://en.wikipedia.org/wiki/Box_plot>`_
    + Strip plots
    + Strip-box plots (strip and box plots overlaid)
    + Strip-histogram plots (strip and histogram plots overlaid)
    
- **Plots without a categorical axis**

    + `Histograms <https://en.wikipedia.org/wiki/Histogram>`_
    + `ECDFs <https://en.wikipedia.org/wiki/Empirical_distribution_function)>`_

In general, you should **plot all of your data**. Strip plots and `ECDFs <https://en.wikipedia.org/wiki/Empirical_distribution_function>`_ do this, but box plots and histograms do not . A box plot provides a quick visual display of summary statistics (specifically quantiles), and a histogram provides an approximate visualization of the probability density function underlying the data generation process. Both serve a purpose, and are included in iqplot. However, in my opinion, a box plot should mostly be used when overlayed with a strip plot, as strip-box plot. The box plot is then an annotation of the plot of all of the data. For visualizing the shape of distributions, ECDFs are superior to histograms in that they serve to visualize the cumulative distribution function of the underlying the data generation process (which has the same information as the probability density function) and do so while plotting all data and not introducing arbitrary binning.

So, you should view the box plot and histogram functionality of iqplot to be add-ons, with strip plots (annotated with boxes and histograms if desired) and ECDFs being the most important functionality.


Why iqplot?
===========

This package was originally developed to enable rapid generation of these plots, particularly ECDFs using `Bokeh <https://bokeh.pydata.org/>`_, a powerful plotting library. `HoloViews <https://holoviews.org/>`_ has emerged as the primary high-level plotting package that can use Bokeh to render plots. Much of what iqplot provides is available in HoloViews, and you can see comparisons in the :ref:`Relationship to HoloViews` section of the iqplot documentation. Nonetheless, I have still found this package useful to quickly generate plots. Importantly, generating ECDFs with bootstrapped confidence intervals is available in iqplot, but is nontrivial to do using HoloViews.


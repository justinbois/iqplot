"""Visualization of how data are distributed, split or colored by a
categorical variable."""

import copy
import warnings

import numpy as np
import pandas as pd

import colorcet

import bokeh.models
import bokeh.plotting

from . import utils


def ecdf(
    data=None,
    q=None,
    cats=None,
    q_axis="x",
    palette=None,
    order=None,
    p=None,
    show_legend=True,
    legend_label=None,
    legend_location="right",
    legend_orientation="vertical",
    tooltips=None,
    complementary=False,
    kind="collection",
    style="dots",
    conf_int=False,
    ptiles=[2.5, 97.5],
    n_bs_reps=10000,
    click_policy="hide",
    marker="circle",
    marker_kwargs=None,
    line_kwargs=None,
    conf_int_kwargs=None,
    horizontal=None,
    val=None,
    **kwargs,
):
    """
    Make an ECDF plot.

    Parameters
    ----------
    data : Pandas DataFrame, 1D Numpy array, or xarray
        DataFrame containing tidy data for plotting.  If a Numpy array,
        a single category is assumed and a box plot generated from
        data.
    q : hashable
        Name of column to use as quantitative variable if `data` is a
        Pandas DataFrame. Otherwise, `q` is used as the quantitative
        axis label.
    cats : hashable or list of hashables
        Name of column(s) to use as categorical variable(s).
    q_axis : str, either 'x' or 'y', default 'x'
        Axis along which the quantitative value varies.
    palette : list of strings of hex colors, or single hex string
        If a list, color palette to use. If a single string representing
        a hex color, all glyphs are colored with that color. Default is
        colorcet.b_glasbey_category10 from the colorcet package.
    order : list or None
        If not None, must be a list of unique group names when the input
        data frame is grouped by `cats`. The order of the list specifies
        the ordering of the categorical variables in the legend. If
        None, the categories appear in the order in which they appeared
        in the inputted data frame.
    p : bokeh.plotting.Figure instance, or None (default)
        If None, create a new figure. Otherwise, populate the existing
        figure `p`.
    show_legend : bool, default False
        If True, display legend.
    legend_label : str, default None
        If `cats` is None and `show_legend` is True, then if
        `legend_label` is not None, a legend is created for the glyph
        on the plot and labeled with `legend_label`. Otherwise, no
        legend is created if `cats` is None.
    legend_location : str, default 'right'
        Location of legend. If one of "right", "left", "above", or
        "below", the legend is placed outside of the plot area. If one
        of "top_left", "top_center", "top_right", "center_right",
        "bottom_right", "bottom_center", "bottom_left", "center_left",
        or "center", the legend is placed within the plot area. If a
        2-tuple, legend is placed according to the coordinates in the
        tuple.
    legend_orientation : str, default 'vertical'
        Either 'horizontal' or 'vertical'.
    tooltips : list of 2-tuples
        Specification for tooltips as per Bokeh specifications. For
        example, if we want `col1` and `col2` tooltips, we can use
        `tooltips=[('label 1': '@col1'), ('label 2': '@col2')]`. Ignored
        if `style` is 'staircase'.
    complementary : bool, default False
        If True, plot the empirical complementary cumulative
        distribution function.
    kind : str, default 'collection'
        If 'collection', the figure is populated with a collection of
        ECDFs coded with colors based on the categorical variables. If
        'colored', the figure is populated with a single ECDF with
        circles colored based on the categorical variables.
    style : str, default 'dots'
        The style of ECDF to make.

            - dots: Each data point is plotted as a dot.
            - staircase: ECDF is plotted as a traditional staircase.
            - formal: Strictly adhere to the definition of an ECDF.
    conf_int : bool, default False
        If True, display confidence interval of ECDF.
    ptiles : list, default [2.5, 97.5]
        The percentiles to use for the confidence interval. Ignored if
        `conf_int` is False.
    n_bs_reps : int, default 1000
        Number of bootstrap replicates to do to compute confidence
        interval. Ignored if `conf_int` is False.
    click_policy : str, default 'hide'
        Either 'hide', 'mute', or None; how the glyphs respond when the
        corresponding category is clicked in the legend.
    marker : str, default 'circle'
        Name of marker to be used in the plot (ignored if `style` is
        'staircase'). Must be one of['asterisk', 'circle',
        'circle_cross', 'circle_x', 'cross', 'dash', 'diamond',
        'diamond_cross', 'hex', 'inverted_triangle', 'square',
        'square_cross', 'square_x', 'triangle', 'x']
    marker_kwargs : dict
        Keyword arguments to be passed to `p.circle()`.
    line_kwargs : dict
        Kwargs to be passed to `p.line()`, `p.ray()`, and `p.segment()`.
    conf_int_kwargs : dict
        kwargs to pass into patches depicting confidence intervals.
    horizontal : bool or None, default None
        Deprecated. Use `q_axis`.
    val : hashable
        Deprecated, use `q`.
    kwargs
        Any kwargs to be passed to `bokeh.plotting.figure()` when making
        the plot.

    Returns
    -------
    output : bokeh.plotting.Figure instance
        Plot populated with ECDFs.
    """
    # Protect against mutability of dicts
    marker_kwargs = copy.copy(marker_kwargs)
    line_kwargs = copy.copy(line_kwargs)
    conf_int_kwargs = copy.copy(conf_int_kwargs)

    q = utils._parse_deprecations(q, q_axis, val, horizontal, "y")

    if style == "formal" and complementary:
        raise NotImplementedError("Complementary formal ECDFs not yet implemented.")

    if palette is None:
        palette = colorcet.b_glasbey_category10

    data, q, cats, show_legend = utils._data_cats(
        data, q, cats, show_legend, legend_label
    )

    cats, cols = utils._check_cat_input(
        data, cats, q, None, None, tooltips, palette, order, marker_kwargs
    )

    kwargs = utils._fig_dimensions(kwargs)

    if conf_int and "y_axis_type" in kwargs and kwargs["y_axis_type"] == "log":
        warnings.warn(
            "Cannot reliably draw confidence intervals with a y-axis on a log scale because zero cannot be represented. Omitting confidence interval."
        )
        conf_int = False
    if (
        conf_int
        and "x_axis_type" in kwargs
        and kwargs["x_axis_type"] == "log"
        and (data[q] <= 0).any()
    ):
        warnings.warn(
            "Cannot draw confidence intervals with a x-axis on a log scale because some values are negative. Any negative values will be omitted from the ECDF."
        )
        conf_int = False

    if marker_kwargs is None:
        marker_kwargs = {}
    if line_kwargs is None:
        line_kwargs = {}

    y = "__ECCDF" if complementary else "__ECDF"

    if q_axis == "y":
        if "x_axis_label" not in kwargs:
            if complementary:
                kwargs["x_axis_label"] = "ECCDF"
            else:
                kwargs["x_axis_label"] = "ECDF"
    else:
        if "y_axis_label" not in kwargs:
            if complementary:
                kwargs["y_axis_label"] = "ECCDF"
            else:
                kwargs["y_axis_label"] = "ECDF"

    if q_axis == "y":
        if "y_axis_label" not in kwargs:
            kwargs["y_axis_label"] = q
    else:
        if "x_axis_label" not in kwargs:
            kwargs["x_axis_label"] = q

    if style in ["formal", "staircase"] and "line_width" not in line_kwargs:
        line_kwargs["line_width"] = 2

    if conf_int_kwargs is None:
        conf_int_kwargs = {}
    if "fill_alpha" not in conf_int_kwargs:
        conf_int_kwargs["fill_alpha"] = 0.5
    if "line_alpha" not in conf_int_kwargs and "line_color" not in conf_int_kwargs:
        conf_int_kwargs["line_alpha"] = 0

    df = data.copy()
    if kind == "collection":
        if style == "dots":
            df[y] = df.groupby(cats)[q].transform(_ecdf_y, complementary=complementary)
    elif kind == "colored":
        df[y] = df[q].transform(_ecdf_y, complementary=complementary)
        cols += [y]
    else:
        raise RuntimeError("`kind` must be in `['collection', 'colored']")

    _, df["__label"] = utils._source_and_labels_from_cats(df, cats)
    cols += ["__label"]

    if order is not None:
        if type(cats) in [list, tuple]:
            df["__sort"] = df.apply(lambda r: order.index(tuple(r[cats])), axis=1)
        else:
            df["__sort"] = df.apply(lambda r: order.index(r[cats]), axis=1)
        df = df.sort_values(by="__sort")

    if p is None:
        p = bokeh.plotting.figure(**kwargs)

    if style == "dots":
        marker_fun = utils._get_marker(p, marker)

    if tooltips is not None:
        if style in ["formal", "staircase"]:
            warnings.warn(
                "Cannot have tooltips for formal ECDFs because there are no points to hover over. Omitting tooltips"
            )
        else:
            p.add_tools(bokeh.models.HoverTool(tooltips=tooltips))

    markers = []
    lines = []
    patches = []
    labels = []

    if kind == "collection":
        # Explicitly loop to enable click policies on the legend
        # (not possible with factors)
        for i, (name, g) in enumerate(df.groupby(cats, sort=False)):
            labels.append(g["__label"].iloc[0])
            if conf_int:
                conf_int_kwargs["fill_color"] = palette[i % len(palette)]
                # conf_int_kwargs["legend_label"] = g["__label"].iloc[0]
                p, patch = _ecdf_conf_int(
                    p,
                    g[q],
                    complementary=complementary,
                    q_axis=q_axis,
                    n_bs_reps=n_bs_reps,
                    ptiles=ptiles,
                    **conf_int_kwargs,
                )
                patches.append(patch)

            marker_kwargs["color"] = palette[i % len(palette)]
            # marker_kwargs["legend_label"] = g["__label"].iloc[0]
            line_kwargs["color"] = palette[i % len(palette)]
            # line_kwargs["legend_label"] = g["__label"].iloc[0]
            if style == "staircase":
                p, new_line = _staircase_ecdf(
                    p,
                    data=g[q],
                    complementary=complementary,
                    q_axis=q_axis,
                    line_kwargs=line_kwargs,
                )
                lines.append(new_line)
            elif style == "dots":
                if q_axis == "y":
                    markers.append(marker_fun(source=g, x=y, y=q, **marker_kwargs))
                else:
                    markers.append(marker_fun(source=g, x=q, y=y, **marker_kwargs))
            elif style == "formal":
                p, circle, segment = _formal_ecdf(
                    p,
                    data=g[q],
                    complementary=complementary,
                    q_axis=q_axis,
                    marker_kwargs=marker_kwargs,
                    line_kwargs=line_kwargs,
                )
                markers.append(circle)
                lines.append(segment)
    elif kind == "colored":
        if style in ["formal", "staircase"]:
            raise RuntimeError(
                "Cannot have a formal or staircase ECDF with `kind='colored'`."
            )

        if conf_int:
            if "fill_color" not in conf_int_kwargs:
                conf_int_kwargs["fill_color"] = "gray"

            p, patch = _ecdf_conf_int(
                p,
                df[q],
                complementary=complementary,
                q_axis=q_axis,
                n_bs_reps=n_bs_reps,
                ptiles=ptiles,
                **conf_int_kwargs,
            )

        y = "__ECCDF" if complementary else "__ECDF"

        # Explicitly loop to enable click policies on the legend (not possible with factors)
        for i, (name, g) in enumerate(df.groupby(cats, sort=False)):
            source = bokeh.models.ColumnDataSource(g[cols])
            mkwargs = marker_kwargs
            # mkwargs["legend_label"] = g["__label"].iloc[0]
            mkwargs["color"] = palette[i % len(palette)]
            labels.append(g["__label"].iloc[0])
            if q_axis == "y":
                markers.append(marker_fun(source=source, x=y, y=q, **mkwargs))
            else:
                markers.append(marker_fun(source=source, x=q, y=y, **mkwargs))

    return _dist_legend(
        p,
        show_legend,
        legend_location,
        legend_orientation,
        click_policy,
        labels,
        markers,
        lines,
        patches,
    )


def histogram(
    data=None,
    q=None,
    cats=None,
    palette=None,
    order=None,
    q_axis="x",
    p=None,
    rug=True,
    rug_height=0.05,
    show_legend=None,
    legend_label=None,
    legend_location="right",
    legend_orientation="vertical",
    bins="freedman-diaconis",
    density=False,
    kind="step_filled",
    click_policy="hide",
    line_kwargs=None,
    fill_kwargs=None,
    rug_kwargs=None,
    horizontal=None,
    val=None,
    **kwargs,
):
    """
    Make a plot of histograms.

    Parameters
    ----------
    data : Pandas DataFrame, 1D Numpy array, or xarray
        DataFrame containing tidy data for plotting.  If a Numpy array,
        a single category is assumed and a box plot generated from
        data.
    q : hashable
        Name of column to use as quantitative variable if `data` is a
        Pandas DataFrame. Otherwise, `q` is used as the quantitative
        axis label.
    cats : hashable or list of hashables
        Name of column(s) to use as categorical variable(s).
    q_axis : str, either 'x' or 'y', default 'x'
        Axis along which the quantitative value varies.
    palette : list of strings of hex colors, or single hex string
        If a list, color palette to use. If a single string representing
        a hex color, all glyphs are colored with that color. Default is
        colorcet.b_glasbey_category10 from the colorcet package.
    order : list or None
        If not None, must be a list of unique group names when the input
        data frame is grouped by `cats`. The order of the list specifies
        the ordering of the categorical variables in the legend. If
        None, the categories appear in the order in which they appeared
        in the inputted data frame.
    p : bokeh.plotting.Figure instance, or None (default)
        If None, create a new figure. Otherwise, populate the existing
        figure `p`.
    legend_label : str, default None
        If `cats` is None and `show_legend` is True, then if
        `legend_label` is not None, a legend is created for the glyph
        on the plot and labeled with `legend_label`. Otherwise, no
        legend is created if `cats` is None.
    legend_location : str, default 'right'
        Location of legend. If one of "right", "left", "above", or
        "below", the legend is placed outside of the plot area. If one
        of "top_left", "top_center", "top_right", "center_right",
        "bottom_right", "bottom_center", "bottom_left", "center_left",
        or "center", the legend is placed within the plot area. If a
        2-tuple, legend is placed according to the coordinates in the
        tuple.
    legend_orientation : str, default 'vertical'
        Either 'horizontal' or 'vertical'.
    bins : int, array_like, or str, default 'freedman-diaconis'
        If int or array_like, setting for `bins` kwarg to be passed to
        `np.histogram()`. If 'exact', then each unique value in the
        data gets its own bin. If 'integer', then integer data is
        assumed and each integer gets its own bin. If 'sqrt', uses the
        square root rule to determine number of bins. If
        `freedman-diaconis`, uses the Freedman-Diaconis rule for number
        of bins.
    rug : bool, default True
        If True, also include a rug plot. If, however, `bins` is 'exact'
        or 'integer', the `rug` kwarg is ignored.
    rug_height : float, default 0.05
        Height of the rug plot as a fraction of the highest point in the
        histograms.
    density : bool, default False
        If True, normalize the histograms. Otherwise, base the
        histograms on counts.
    kind : str, default 'step_filled'
        The kind of histogram to display. Allowed values are 'step' and
        'step_filled'.
    click_policy : str, default 'hide'
        Either 'hide', 'mute', or None; how the glyphs respond when the
        corresponding category is clicked in the legend.
    line_kwargs : dict
        Keyword arguments to pass to `p.line()` in constructing the
        histograms. By default, {"line_width": 2}.
    fill_kwargs : dict
        Keyword arguments to pass to `p.patch()` when making the fill
        for the step-filled histogram. Ignored if `kind = 'step'`. By
        default {"fill_alpha": 0.3, "line_alpha": 0}.
    rug_kwargs : dict
        Keyword arguments to pass to `p.multi_line()` when making the
        rug plot.
    horizontal : bool or None, default None
        Deprecated. Use `q_axis`.
    val : hashable
        Deprecated, use `q`.
    kwargs
        Any kwargs to be passed to `bokeh.plotting.figure()` when making
        the plot.

    Returns
    -------
    output : Bokeh figure
        Figure populated with histograms.
    """
    # Protect against mutability of dicts
    line_kwargs = copy.copy(line_kwargs)
    fill_kwargs = copy.copy(fill_kwargs)
    rug_kwargs = copy.copy(rug_kwargs)

    if type(bins) == str and bins in ["integer", "exact"]:
        rug = False

    q = utils._parse_deprecations(q, q_axis, val, horizontal, "y")

    if palette is None:
        palette = colorcet.b_glasbey_category10

    df, q, cats, show_legend = utils._data_cats(
        data, q, cats, show_legend, legend_label
    )

    if show_legend is None:
        if cats is None:
            show_legend = False
        else:
            show_legend = True

    if type(bins) == str and bins not in [
        "integer",
        "exact",
        "sqrt",
        "freedman-diaconis",
    ]:
        raise RuntimeError("Invalid bin specification.")

    if cats is None:
        df["__cat"] = "__dummy_cat"
        if show_legend:
            raise RuntimeError("No legend to show if `cats` is None.")
        if order is not None:
            raise RuntimeError("No `order` is allowed if `cats` is None.")
        cats = "__cat"

    cats, cols = utils._check_cat_input(
        df, cats, q, None, None, None, palette, order, kwargs
    )

    kwargs = utils._fig_dimensions(kwargs)

    if line_kwargs is None:
        line_kwargs = {"line_width": 2}
    if fill_kwargs is None:
        fill_kwargs = {}
    if "fill_alpha" not in fill_kwargs:
        fill_kwargs["fill_alpha"] = 0.3
    if "line_alpha" not in fill_kwargs:
        fill_kwargs["line_alpha"] = 0

    _, df["__label"] = utils._source_and_labels_from_cats(df, cats)
    cols += ["__label"]

    if order is not None:
        if type(cats) in [list, tuple]:
            df["__sort"] = df.apply(lambda r: order.index(tuple(r[cats])), axis=1)
        else:
            df["__sort"] = df.apply(lambda r: order.index(r[cats]), axis=1)
        df = df.sort_values(by="__sort")

    if type(bins) == str and bins == "exact":
        a = np.unique(df[q])
        if len(a) == 1:
            bins = np.array([a[0] - 0.5, a[0] + 0.5])
        else:
            bins = np.concatenate(
                (
                    (a[0] - (a[1] - a[0]) / 2,),
                    (a[1:] + a[:-1]) / 2,
                    (a[-1] + (a[-1] - a[-2]) / 2,),
                )
            )
    elif type(bins) == str and bins == "integer":
        if np.any(df[q] != np.round(df[q])):
            raise RuntimeError("'integer' bins chosen, but data are not integer.")
        bins = np.arange(df[q].min() - 1, df[q].max() + 1) + 0.5

    if p is None:
        kwargs = utils._fig_dimensions(kwargs)

        if "x_axis_label" not in kwargs:
            kwargs["x_axis_label"] = q

        if "y_axis_label" not in kwargs:
            if density:
                kwargs["y_axis_label"] = "density"
            else:
                kwargs["y_axis_label"] = "count"
        if "y_range" not in kwargs:
            kwargs["y_range"] = bokeh.models.DataRange1d(start=0)

        p = bokeh.plotting.figure(**kwargs)

    # Explicitly loop to enable click policies on the legend (not possible with factors)
    max_height = 0
    lines = []
    labels = []
    patches = []
    for i, (name, g) in enumerate(df.groupby(cats, sort=False)):
        e0, f0 = _compute_histogram(g[q], bins, density)

        max_height = max(f0.max(), max_height)

        line_kwargs["color"] = palette[i % len(palette)]

        if q_axis == "y":
            lines.append(p.line(f0, e0, **line_kwargs))
        else:
            lines.append(p.line(e0, f0, **line_kwargs))
        labels.append(g["__label"].iloc[0])

        if kind == "step_filled":
            x2 = [e0.min(), e0.max()]
            y2 = [0, 0]
            fill_kwargs["color"] = palette[i % len(palette)]
            if q_axis == "y":
                p, patch = utils._fill_between(p, f0, e0, y2, x2, **fill_kwargs)
            else:
                p, patch = utils._fill_between(p, e0, f0, x2, y2, **fill_kwargs)
            patches.append(patch)

    # Put in the rug plot
    if rug:
        if rug_kwargs is None:
            rug_kwargs = dict(alpha=0.5, line_width=0.5)
        elif type(rug_kwargs) != dict:
            raise RuntimeError("`rug_kwargs` must be a dictionary.")
        if "alpha" not in rug_kwargs and "line_alpha" not in rug_kwargs:
            rug_kwargs["alpha"] = 0.5
        if "line_width" not in rug_kwargs:
            rug_kwargs["line_width"] = 0.5

        y = [0, max_height * rug_height]

        for i, (name, g) in enumerate(df.groupby(cats, sort=False)):
            xs = [[q_val, q_val] for q_val in g[q].values]
            ys = [y] * len(g)
            if "color" not in rug_kwargs and "line_color" not in rug_kwargs:
                p.multi_line(xs, ys, color=palette[i % len(palette)], **rug_kwargs)
            else:
                p.multi_line(xs, ys, **rug_kwargs)

    return _dist_legend(
        p,
        show_legend,
        legend_location,
        legend_orientation,
        click_policy,
        labels,
        [],
        lines,
        patches,
    )


def _staircase_ecdf(p, data, complementary=False, q_axis="x", line_kwargs={}):
    """
    Create a plot of an ECDF.

    Parameters
    ----------
    p : bokeh.plotting.Figure instance, or None (default)
        If None, create a new figure. Otherwise, populate the existing
        figure `p`.
    data : array_like
        One-dimensional array of data. Nan's are ignored.
    complementary : bool, default False
        If True, plot the empirical complementary cumulative
        distribution functon.
    q_axis : str, default 'x'
        Which axis has the quantitative variable.
    line_kwargs : dict
        kwargs to be passed into p.line and p.ray.

    Returns
    -------
    output : bokeh.plotting.Figure instance
        Plot populated with ECDF.
    """
    # Extract data
    data = utils._convert_data(data)

    # Data points on ECDF
    x, y = _ecdf_vals(data, True, complementary)

    # Line of steps
    if q_axis == "y":
        line = p.line(y, x, **line_kwargs)
    elif q_axis == "x":
        line = p.line(x, y, **line_kwargs)

    # Rays for ends
    if q_axis == "y":
        if complementary:
            p.ray(1, x[0], None, -np.pi / 2, **line_kwargs)
            p.ray(0, x[-1], None, np.pi / 2, **line_kwargs)
        else:
            p.ray(0, x[0], None, -np.pi / 2, **line_kwargs)
            p.ray(1, x[-1], None, np.pi / 2, **line_kwargs)
    elif q_axis == "x":
        if complementary:
            p.ray(x[0], 1, None, np.pi, **line_kwargs)
            p.ray(x[-1], 0, None, 0, **line_kwargs)
        else:
            p.ray(x[0], 0, None, np.pi, **line_kwargs)
            p.ray(x[-1], 1, None, 0, **line_kwargs)

    return p, line


def _formal_ecdf(
    p, data, complementary=False, q_axis="x", marker_kwargs={}, line_kwargs={}
):
    """
    Create a plot of an ECDF.

    Parameters
    ----------
    p : bokeh.plotting.Figure instance, or None (default)
        If None, create a new figure. Otherwise, populate the existing
        figure `p`.
    data : array_like
        One-dimensional array of data. Nan's are ignored.
    complementary : bool, default False
        If True, plot the empirical complementary cumulative
        distribution functon.
    marker_kwargs : dict
        Any kwargs to be passed to p.circle().
    line_kwargs : dict
        Any kwargs to be passed to p.segment() and p.ray().

    Returns
    -------
    output : bokeh.plotting.Figure instance
        Plot populated with ECDF.
    """
    # Extract data
    data = utils._convert_data(data)

    # Data points on ECDF
    x, y = _ecdf_vals(data, complementary)

    # Copy of marker kwargs for unfilled points
    unfilled_kwargs = marker_kwargs.copy()
    unfilled_kwargs["fill_color"] = "white"

    if q_axis == "y":
        segment = p.segment(y[:-1], x[:-1], y[1:], x[:-1], **line_kwargs)
        p.ray(0, x[0], angle=-np.pi / 2, length=0, **line_kwargs)
        p.ray(1, x[-1], angle=np.pi / 2, length=0, **line_kwargs)
        circle = p.circle(y, x, **marker_kwargs)
        p.circle([0], [0], **unfilled_kwargs)
        p.circle(y[:-1], x[1:], **unfilled_kwargs)
    elif q_axis == "x":
        segment = p.segment(x[:-1], y[:-1], x[1:], y[:-1], **line_kwargs)
        p.ray(x[0], 0, angle=np.pi, length=0, **line_kwargs)
        p.ray(x[-1], 1, angle=0, length=0, **line_kwargs)
        circle = p.circle(x, y, **marker_kwargs)
        p.circle([0], [0], **unfilled_kwargs)
        p.circle(x[1:], y[:-1], **unfilled_kwargs)

    return p, circle, segment


def _ecdf_vals(data, staircase=False, complementary=False):
    """Get x, y, values of an ECDF for plotting.
    Parameters
    ----------
    data : ndarray
        One dimensional Numpy array with data.
    staircase : bool, default False
        If True, generate x and y values for ECDF (staircase). If
        False, generate x and y values for ECDF as dots.
    complementary : bool
        If True, return values for ECCDF.

    Returns
    -------
    x : ndarray
        x-values for plot
    y : ndarray
        y-values for plot
    """
    x = np.sort(data)
    y = np.arange(1, len(data) + 1) / len(data)

    if staircase:
        x, y = _to_staircase(x, y)
        if complementary:
            y = 1 - y
    elif complementary:
        y = 1 - y + 1 / len(y)

    return x, y


def _to_staircase(x, y):
    """Convert to formal ECDF."""
    # Set up output arrays
    x_staircase = np.empty(2 * len(x))
    y_staircase = np.empty(2 * len(x))

    # y-values for steps
    y_staircase[0] = 0
    y_staircase[1::2] = y
    y_staircase[2::2] = y[:-1]

    # x- values for steps
    x_staircase[::2] = x
    x_staircase[1::2] = x

    return x_staircase, y_staircase


def _ecdf_conf_int(
    p,
    data,
    complementary=False,
    q_axis="x",
    n_bs_reps=1000,
    ptiles=[2.5, 97.5],
    **kwargs,
):
    """Add an ECDF confidence interval to a plot.

    This method of computing a confidence interval can be thought of as
    computing confidence intervals of the *inverse* ECDF in the sense
    that we compute a confidence interval for the x-values for each of
    the discrete values of the ECDF. This is equivalent to computing
    bootstrap confidence intervals for the ECDF. Here is why.

    Imagine we draw bootstrap samples and for each we make an ECDF.
    Let's say we make 5 such ECDFs and we wish to compute a 60%
    confidence interval. (You can generalize to arbitrary number of
    ECDFs and confidence interval.)

    Each of these 5 ECDFs can be defined as starting at the same point
    and ending at the same point. Specifically, they start at
    x = min(data), y = 0 and end at x = max(data), y = 1. Furthermore,
    they are all monotonically increasing functions.

    Now, let's say we are constructing a confidence interval for the
    ECDF at position x. To do so, we put a dot on the second ECDF from
    the top at x and a dot on the second ECDF from the bottom. This
    gives us the middle 60% of ECDF values.

    Now, say we are constructing a confidence interval for the IECDF. We
    go to ECDF value y and we find the second ECDF from the left and
    place a dot on it. We also put a dot on the second ECDF from the
    right.

    Because all ECDFs are monotonic and start and end at the same
    points, the dot we put on the second-leftmost ECDF is also on the
    second curve from the top for some other x. Similarly, the
    second-rightmost ECDF is also on the second curve from the bottom
    for some other x. (You can sketch this out, and it becomes clear.)

    So, any dot we put on an ECDF for computing a confidence interval
    for an IECDF is also a dot we would put on an ECDF for computing a
    confidence  of the ECDF. If we want to compute the confidence
    interval over the whole domain of x-values, we will cover the same
    set of points if we compute the confidence interval of the ECDF or
    the IECDF. So, we end up filling between the same two sets of
    curves.

    It turns out that the IECDF formulation is actually much easier to
    implement.
    """
    data = utils._convert_data(data)

    bs_reps = np.array(
        [np.sort(np.random.choice(data, size=len(data))) for _ in range(n_bs_reps)]
    )

    # Compute the confidence intervals
    iecdf_low, iecdf_high = np.percentile(np.array(bs_reps), ptiles, axis=0)

    # y-values for ECDFs
    y = np.arange(1, len(data) + 1) / len(data)

    # Make them staircases
    x_low, y_plot = _to_staircase(x=iecdf_low, y=y)
    x_high, _ = _to_staircase(x=iecdf_high, y=y)

    if q_axis == "y":
        if complementary:
            p, patch = utils._fill_between(
                p, x1=1 - y_plot, y1=x_low, x2=1 - y_plot, y2=x_high, **kwargs
            )
        else:
            p, patch = utils._fill_between(
                p, x1=y_plot, y1=x_low, x2=y_plot, y2=x_high, **kwargs
            )
    elif q_axis == "x":
        if complementary:
            p, patch = utils._fill_between(
                p, x1=x_low, y1=1 - y_plot, x2=x_high, y2=1 - y_plot, **kwargs
            )
        else:
            p, patch = utils._fill_between(
                p, x1=x_low, y1=y_plot, x2=x_high, y2=y_plot, **kwargs
            )
    else:
        raise RuntimeError("`q_axis` must be either 'x' or 'y'.")

    return p, patch


def _ecdf_y(data, complementary=False):
    """Give y-values of an ECDF for an unsorted column in a data frame.

    Parameters
    ----------
    data : Pandas Series
        Series (or column of a DataFrame) from which to generate ECDF
        values
    complementary : bool, default False
        If True, give the ECCDF values.

    Returns
    -------
    output : Pandas Series
        Corresponding y-values for an ECDF when plotted with dots.

    Notes
    -----
    .. This only works for plotting an ECDF with points, not for formal
       or staircase ECDFs
    """
    if complementary:
        return 1 - data.rank(method="first") / len(data) + 1 / len(data)
    else:
        return data.rank(method="first") / len(data)


def _dist_legend(
    p,
    show_legend,
    legend_location,
    legend_orientation,
    click_policy,
    labels,
    markers,
    lines,
    patches,
):
    """Add a legend to a histogram or ECDF plot.
    """
    if show_legend:
        if len(markers) > 0:
            if len(lines) > 0:
                if len(patches) > 0:
                    items = [
                        (label, [marker, line, patch])
                        for label, marker, line, patch in zip(
                            labels, markers, lines, patches
                        )
                    ]
                else:
                    items = [
                        (label, [marker, line])
                        for label, marker, line in zip(labels, lines, markers)
                    ]
            else:
                if len(patches) > 0:
                    items = [
                        (label, [marker, patch])
                        for label, marker, patch in zip(labels, markers, patches)
                    ]
                else:
                    items = [
                        (label, [marker]) for label, marker in zip(labels, markers)
                    ]
        else:
            if len(patches) > 0:
                items = [
                    (label, [line, patch])
                    for label, line, patch in zip(labels, lines, patches)
                ]
            else:
                items = [(label, [line]) for label, line in zip(labels, lines)]

        if len(p.legend) == 1:
            for item in items:
                p.legend.items.append(
                    bokeh.models.LegendItem(label=item[0], renderers=item[1])
                )
        else:
            if len(p.legend) > 1:
                warnings.warn(
                    "Ambiguous which legend to add glyphs to. Creating new legend."
                )
            if legend_location in ["right", "left", "above", "below"]:
                legend = bokeh.models.Legend(
                    items=items, location="center", orientation=legend_orientation
                )
                p.add_layout(legend, legend_location)
            elif (
                legend_location
                in [
                    "top_left",
                    "top_center",
                    "top_right",
                    "center_right",
                    "bottom_right",
                    "bottom_center",
                    "bottom_left",
                    "center_left",
                    "center",
                ]
                or type(legend_location) == tuple
            ):
                legend = bokeh.models.Legend(
                    items=items,
                    location=legend_location,
                    orientation=legend_orientation,
                )
                p.add_layout(legend, "center")
            else:
                raise RuntimeError(
                    'Invalid `legend_location`. Must be a 2-tuple specifying location or one of ["right", "left", "above", "below", "top_left", "top_center", "top_right", "center_right", "bottom_right", "bottom_center", "bottom_left", "center_left", "center"]'
                )

        p.legend.click_policy = click_policy

    return p


def _compute_histogram(data, bins, density):
    if type(bins) == str and bins == "sqrt":
        bins = int(np.ceil(np.sqrt(len(data))))
    elif type(bins) == str and bins == "freedman-diaconis":
        h = 2 * (np.percentile(data, 75) - np.percentile(data, 25)) / np.cbrt(len(data))
        if h == 0.0:
            bins = 3
        else:
            bins = int(np.ceil((data.max() - data.min()) / h))

    f, e = np.histogram(data, bins=bins, density=density)
    e0 = np.empty(2 * len(e))
    f0 = np.empty(2 * len(e))
    e0[::2] = e
    e0[1::2] = e
    f0[0] = 0
    f0[-1] = 0
    f0[1:-1:2] = f
    f0[2:-1:2] = f

    return e0, f0

import copy
import warnings

import numpy as np
import pandas as pd

import colorcet

import bokeh.models
import bokeh.plotting

from . import utils


def strip(
    data=None,
    q=None,
    cats=None,
    q_axis="x",
    palette=None,
    order=None,
    p=None,
    show_legend=False,
    color_column=None,
    parcoord_column=None,
    tooltips=None,
    marker="circle",
    jitter=False,
    marker_kwargs=None,
    jitter_kwargs=None,
    parcoord_kwargs=None,
    horizontal=None,
    val=None,
    **kwargs,
):
    """
    Make a strip plot.

    Parameters
    ----------
    data : Pandas DataFrame, 1D Numpy array, or xarray
        DataFrame containing tidy data for plotting.  If a Numpy array,
        a single category is assumed and a strip plot generated from
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
        the ordering of the categorical variables on the categorical
        axis and legend. If None, the categories appear in the order in
        which they appeared in the inputted data frame.
    p : bokeh.plotting.Figure instance, or None (default)
        If None, create a new figure. Otherwise, populate the existing
        figure `p`.
    show_legend : bool, default False
        If True, display legend.
    color_column : hashable, default None
        Column of `data` to use in determining color of glyphs. If None,
        then `cats` is used.
    parcoord_column : hashable, default None
        Column of `data` to use to construct a parallel coordinate plot.
        Data points with like entries in the parcoord_column are
        connected with lines.
    tooltips : list of 2-tuples
        Specification for tooltips as per Bokeh specifications. For
        example, if we want `col1` and `col2` tooltips, we can use
        `tooltips=[('label 1': '@col1'), ('label 2': '@col2')]`.
    marker : str, default 'circle'
        Name of marker to be used in the plot (ignored if `formal` is
        False). Must be one of['asterisk', 'circle', 'circle_cross',
        'circle_x', 'cross', 'dash', 'diamond', 'diamond_cross', 'hex',
        'inverted_triangle', 'square', 'square_cross', 'square_x',
        'triangle', 'x']
    jitter : bool, default False
        If True, apply a jitter transform to the glyphs.
    marker_kwargs : dict
        Keyword arguments to pass when adding markers to the plot.
        ["x", "y", "source", "cat", "legend"] are note allowed because
        they are determined by other inputs.
    jitter_kwargs : dict
        Keyword arguments to be passed to `bokeh.transform.jitter()`. If
        not specified, default is
        `{'distribution': 'normal', 'width': 0.1}`. If the user
        specifies `{'distribution': 'uniform'}`, the `'width'` entry is
        adjusted to 0.4.
    horizontal : bool or None, default None
        Deprecated. Use `q_axis`.
    val : hashable
        Deprecated, use `q`.
    kwargs
        Any kwargs to be passed to `bokeh.plotting.figure()` when
        instantiating the figure.

    Returns
    -------
    output : bokeh.plotting.Figure instance
        Plot populated with a strip plot.
    """
    # Protect against mutability of dicts
    jitter_kwargs = copy.copy(jitter_kwargs)
    marker_kwargs = copy.copy(marker_kwargs)

    q = utils._parse_deprecations(q, q_axis, val, horizontal, "x")

    if palette is None:
        palette = colorcet.b_glasbey_category10

    data, q, cats, show_legend = utils._data_cats(data, q, cats, show_legend, None)

    cats, cols = utils._check_cat_input(
        data, cats, q, color_column, parcoord_column, tooltips, palette, order, kwargs
    )

    grouped = data.groupby(cats, sort=False)

    if p is None:
        p, factors, color_factors = _cat_figure(
            data, grouped, q, order, color_column, q_axis, kwargs
        )
    else:
        if type(p.x_range) == bokeh.models.ranges.FactorRange and q_axis == "x":
            raise RuntimeError("`q_axis` is 'x', but `p` has a categorical x-axis.")
        elif type(p.y_range) == bokeh.models.ranges.FactorRange and q_axis == "y":
            raise RuntimeError("`q_axis` is 'y', but `p` has a categorical y-axis.")

        _, factors, color_factors = _get_cat_range(
            data, grouped, order, color_column, q_axis
        )

    if tooltips is not None:
        p.add_tools(bokeh.models.HoverTool(tooltips=tooltips))

    if jitter_kwargs is None:
        jitter_kwargs = dict(width=0.1, mean=0, distribution="normal")
    elif type(jitter_kwargs) != dict:
        raise RuntimeError("`jitter_kwargs` must be a dict.")
    elif "width" not in jitter_kwargs:
        if (
            "distribution" not in jitter_kwargs
            or jitter_kwargs["distribution"] == "uniform"
        ):
            jitter_kwargs["width"] = 0.4
        else:
            jitter_kwargs["width"] = 0.1

    if marker_kwargs is None:
        marker_kwargs = {}
    elif type(marker_kwargs) != dict:
        raise RuntimeError("`marker_kwargs` must be a dict.")

    if "color" not in marker_kwargs:
        if color_column is None:
            color_column = "cat"
        marker_kwargs["color"] = bokeh.transform.factor_cmap(
            color_column, palette=palette, factors=color_factors
        )

    if marker == "tick":
        marker = "dash"
    marker_fun = utils._get_marker(p, marker)

    if marker == "dash":
        if "angle" not in marker_kwargs and q_axis == "x":
            marker_kwargs["angle"] = np.pi / 2
        if "size" not in marker_kwargs:
            if q_axis == "x":
                marker_kwargs["size"] = p.plot_height * 0.25 / len(grouped)
            else:
                marker_kwargs["size"] = p.plot_width * 0.25 / len(grouped)

    source = _cat_source(data, cats, cols, color_column)

    if show_legend and "legend_field" not in marker_kwargs:
        marker_kwargs["legend_field"] = "__label"

    if q_axis == "x":
        x = q
        if jitter:
            jitter_kwargs["range"] = p.y_range
            y = bokeh.transform.jitter("cat", **jitter_kwargs)
        else:
            y = "cat"
        p.ygrid.grid_line_color = None
    else:
        y = q
        if jitter:
            jitter_kwargs["range"] = p.x_range
            x = bokeh.transform.jitter("cat", **jitter_kwargs)
        else:
            x = "cat"
        p.xgrid.grid_line_color = None

    if parcoord_column is not None:
        source_pc = _parcoord_source(data, q, cats, q_axis, parcoord_column, factors)

        if parcoord_kwargs is None:
            line_color = "gray"
            parcoord_kwargs = {}
        elif type(parcoord_kwargs) != dict:
            raise RuntimeError("`parcoord_kwargs` must be a dict.")

        if "color" in parcoord_kwargs and "line_color" not in parcoord_kwargs:
            line_color = parcoord_kwargs.pop("color")
        else:
            line_color = parcoord_kwargs.pop("line_color", "gray")

        p.multi_line(
            source=source_pc, xs="xs", ys="ys", line_color=line_color, **parcoord_kwargs
        )

    marker_fun(source=source, x=x, y=y, **marker_kwargs)

    return p


def box(
    data=None,
    q=None,
    cats=None,
    q_axis="x",
    palette=None,
    order=None,
    p=None,
    whisker_caps=False,
    display_points=True,
    outlier_marker="circle",
    min_data=5,
    box_kwargs=None,
    median_kwargs=None,
    whisker_kwargs=None,
    outlier_kwargs=None,
    display_outliers=None,
    horizontal=None,
    val=None,
    **kwargs,
):
    """
    Make a box-and-whisker plot.

    Parameters
    ----------
    data : Pandas DataFrame, 1D Numpy array, or xarray
        DataFrame containing tidy data for plotting.  If a Numpy array,
        a single category is assumed and a box plot with a single box is
        generated from data.
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
        the ordering of the categorical variables on the categorical
        axis and legend. If None, the categories appear in the order in
        which they appeared in the inputted data frame.
    p : bokeh.plotting.Figure instance, or None (default)
        If None, create a new figure. Otherwise, populate the existing
        figure `p`.
    whisker_caps : bool, default False
        If True, put caps on whiskers. If False, omit caps.
    display_points : bool, default True
        If True, display outliers and any other points that arise from
        categories with fewer than `min_data` data points; otherwise
        suppress them. This should only be False when using the boxes
        as annotation on another plot.
    outlier_marker : str, default 'circle'
        Name of marker to be used in the plot (ignored if `formal` is
        False). Must be one of['asterisk', 'circle', 'circle_cross',
        'circle_x', 'cross', 'dash', 'diamond', 'diamond_cross', 'hex',
        'inverted_triangle', 'square', 'square_cross', 'square_x',
        'triangle', 'x']
    min_data : int, default 5
        Minimum number of data points in a given category in order to
        make a box and whisker. Otherwise, individual data points are
        plotted as in a strip plot.
    box_kwargs : dict, default None
        A dictionary of kwargs to be passed into `p.hbar()` or
        `p.vbar()` when constructing the boxes for the box plot.
    median_kwargs : dict, default None
        A dictionary of kwargs to be passed into `p.hbar()` or
        `p.vbar()` when constructing the median line for the box plot.
    whisker_kwargs : dict, default None
        A dictionary of kwargs to be passed into `p.segment()`
        when constructing the whiskers for the box plot.
    outlier_kwargs : dict, default None
        A dictionary of kwargs to be passed into `p.circle()`
        when constructing the outliers for the box plot.
    horizontal : bool or None, default None
        Deprecated. Use `q_axis`.
    val : hashable
        Deprecated, use `q`.
    kwargs
        Kwargs that are passed to bokeh.plotting.figure() in contructing
        the figure.

    Returns
    -------
    output : bokeh.plotting.Figure instance
        Plot populated with box-and-whisker plot.

    Notes
    -----
    Uses the Tukey convention for box plots. The top and bottom of
    the box are respectively the 75th and 25th percentiles of the
    data. The line in the middle of the box is the median. The top
    whisker extends to the maximum of the set of data points that are
    less than 1.5 times the IQR beyond the top of the box, with an
    analogous definition for the lower whisker. Data points not
    between the ends of the whiskers are considered outliers and are
    plotted as individual points.
    """
    # Protect against mutability of dicts
    box_kwargs = copy.copy(box_kwargs)
    median_kwargs = copy.copy(median_kwargs)
    whisker_kwargs = copy.copy(whisker_kwargs)
    outlier_kwargs = copy.copy(outlier_kwargs)

    q = utils._parse_deprecations(q, q_axis, val, horizontal, "x")

    if display_outliers is not None:
        warnings.warn(
            f"`display_outliers` is deprecated. Use `display_points`. Using `display_points={display_outliers}.",
            DeprecationWarning,
        )
        display_points = display_outliers

    if palette is None:
        palette = colorcet.b_glasbey_category10

    data, q, cats, _ = utils._data_cats(data, q, cats, False, None)

    cats, cols = utils._check_cat_input(
        data, cats, q, None, None, None, palette, order, box_kwargs
    )

    if outlier_kwargs is None:
        outlier_kwargs = dict()
    elif type(outlier_kwargs) != dict:
        raise RuntimeError("`outlier_kwargs` must be a dict.")

    if box_kwargs is None:
        box_kwargs = {"line_color": None}
        box_width = 0.4
    elif type(box_kwargs) != dict:
        raise RuntimeError("`box_kwargs` must be a dict.")
    else:
        box_width = box_kwargs.pop("width", 0.4)
        if "line_color" not in box_kwargs:
            box_kwargs["line_color"] = None

    if whisker_kwargs is None:
        if "fill_color" in box_kwargs:
            whisker_kwargs = {"line_color": box_kwargs["fill_color"]}
        else:
            whisker_kwargs = {"line_color": "black"}
    elif type(whisker_kwargs) != dict:
        raise RuntimeError("`whisker_kwargs` must be a dict.")

    if median_kwargs is None:
        median_kwargs = {"line_color": "white"}
    elif type(median_kwargs) != dict:
        raise RuntimeError("`median_kwargs` must be a dict.")
    elif "line_color" not in median_kwargs:
        median_kwargs["line_color"] = white

    if q_axis == "x":
        if "height" in box_kwargs:
            warnings.warn("'height' entry in `box_kwargs` ignored; using `box_width`.")
            del box_kwargs["height"]
    else:
        if "width" in box_kwargs:
            warnings.warn("'width' entry in `box_kwargs` ignored; using `box_width`.")
            del box_kwargs["width"]

    grouped = data.groupby(cats, sort=False)

    if p is None:
        p, factors, color_factors = _cat_figure(
            data, grouped, q, order, None, q_axis, kwargs
        )
    else:
        _, factors, color_factors = _get_cat_range(data, grouped, order, None, q_axis)

    marker_fun = utils._get_marker(p, outlier_marker)

    source_box, source_outliers = _box_source(data, cats, q, cols, min_data)

    if "color" in outlier_kwargs:
        if "line_color" in outlier_kwargs or "fill_color" in outlier_kwargs:
            raise RuntimeError(
                "If `color` is in `outlier_kwargs`, `line_color` and `fill_color` cannot be."
            )
    else:
        if "fill_color" in box_kwargs:
            if "fill_color" not in outlier_kwargs:
                outlier_kwargs["fill_color"] = box_kwargs["fill_color"]
            if "line_color" not in outlier_kwargs:
                outlier_kwargs["line_color"] = box_kwargs["fill_color"]
        else:
            if "fill_color" not in outlier_kwargs:
                outlier_kwargs["fill_color"] = bokeh.transform.factor_cmap(
                    "cat", palette=palette, factors=factors
                )
            if "line_color" not in outlier_kwargs:
                outlier_kwargs["line_color"] = bokeh.transform.factor_cmap(
                    "cat", palette=palette, factors=factors
                )

    if "fill_color" not in box_kwargs:
        box_kwargs["fill_color"] = bokeh.transform.factor_cmap(
            "cat", palette=palette, factors=factors
        )

    if q_axis == "x":
        p.segment(
            source=source_box,
            y0="cat",
            y1="cat",
            x0="top",
            x1="top_whisker",
            **whisker_kwargs,
        )
        p.segment(
            source=source_box,
            y0="cat",
            y1="cat",
            x0="bottom",
            x1="bottom_whisker",
            **whisker_kwargs,
        )
        if whisker_caps:
            p.hbar(
                source=source_box,
                y="cat",
                left="top_whisker",
                right="top_whisker",
                height=box_width / 4,
                **whisker_kwargs,
            )
            p.hbar(
                source=source_box,
                y="cat",
                left="bottom_whisker",
                right="bottom_whisker",
                height=box_width / 4,
                **whisker_kwargs,
            )
        p.hbar(
            source=source_box,
            y="cat",
            left="bottom",
            right="top",
            height=box_width,
            **box_kwargs,
        )
        p.hbar(
            source=source_box,
            y="cat",
            left="middle",
            right="middle",
            height=box_width,
            **median_kwargs,
        )
        if display_points:
            marker_fun(source=source_outliers, y="cat", x=q, **outlier_kwargs)
        p.ygrid.grid_line_color = None
    else:
        p.segment(
            source=source_box,
            x0="cat",
            x1="cat",
            y0="top",
            y1="top_whisker",
            **whisker_kwargs,
        )
        p.segment(
            source=source_box,
            x0="cat",
            x1="cat",
            y0="bottom",
            y1="bottom_whisker",
            **whisker_kwargs,
        )
        if whisker_caps:
            p.vbar(
                source=source_box,
                x="cat",
                bottom="top_whisker",
                top="top_whisker",
                width=box_width / 4,
                **whisker_kwargs,
            )
            p.vbar(
                source=source_box,
                x="cat",
                bottom="bottom_whisker",
                top="bottom_whisker",
                width=box_width / 4,
                **whisker_kwargs,
            )
        p.vbar(
            source=source_box,
            x="cat",
            bottom="bottom",
            top="top",
            width=box_width,
            **box_kwargs,
        )
        p.vbar(
            source=source_box,
            x="cat",
            bottom="middle",
            top="middle",
            width=box_width,
            **median_kwargs,
        )
        if display_points:
            marker_fun(source=source_outliers, x="cat", y=q, **outlier_kwargs)
        p.xgrid.grid_line_color = None

    return p


def stripbox(
    data=None,
    q=None,
    cats=None,
    q_axis="x",
    palette=None,
    order=None,
    p=None,
    show_legend=False,
    top_level="strip",
    color_column=None,
    parcoord_column=None,
    tooltips=None,
    marker="circle",
    jitter=False,
    marker_kwargs=None,
    jitter_kwargs=None,
    parcoord_kwargs=None,
    whisker_caps=True,
    display_points=True,
    min_data=5,
    box_kwargs=None,
    median_kwargs=None,
    whisker_kwargs=None,
    horizontal=None,
    val=None,
    **kwargs,
):
    """
    Make a strip plot with a box plot as annotation.

    Parameters
    ----------
    data : Pandas DataFrame, 1D Numpy array, or xarray
        DataFrame containing tidy data for plotting.  If a Numpy array,
        a single category is assumed and a strip plot generated from
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
        the ordering of the categorical variables on the categorical
        axis and legend. If None, the categories appear in the order in
        which they appeared in the inputted data frame.
    p : bokeh.plotting.Figure instance, or None (default)
        If None, create a new figure. Otherwise, populate the existing
        figure `p`.
    top_level : str, default 'strip'
        If 'box', the box plot is overlaid. If 'strip', the strip plot
        is overlaid.
    show_legend : bool, default False
        If True, display legend.
    color_column : hashable, default None
        Column of `data` to use in determining color of glyphs. If None,
        then `cats` is used.
    parcoord_column : hashable, default None
        Column of `data` to use to construct a parallel coordinate plot.
        Data points with like entries in the parcoord_column are
        connected with lines in the strip plot.
    tooltips : list of 2-tuples
        Specification for tooltips as per Bokeh specifications. For
        example, if we want `col1` and `col2` tooltips, we can use
        `tooltips=[('label 1': '@col1'), ('label 2': '@col2')]`.
    marker : str, default 'circle'
        Name of marker to be used in the plot (ignored if `formal` is
        False). Must be one of['asterisk', 'circle', 'circle_cross',
        'circle_x', 'cross', 'dash', 'diamond', 'diamond_cross', 'hex',
        'inverted_triangle', 'square', 'square_cross', 'square_x',
        'triangle', 'x']
    jitter : bool, default False
        If True, apply a jitter transform to the glyphs.
    marker_kwargs : dict
        Keyword arguments to pass when adding markers to the plot.
        ["x", "y", "source", "cat", "legend"] are note allowed because
        they are determined by other inputs.
    jitter_kwargs : dict
        Keyword arguments to be passed to `bokeh.transform.jitter()`. If
        not specified, default is
        `{'distribution': 'normal', 'width': 0.1}`. If the user
        specifies `{'distribution': 'uniform'}`, the `'width'` entry is
        adjusted to 0.4.
    whisker_caps : bool, default True
        If True, put caps on whiskers. If False, omit caps.
    min_data : int, default 5
        Minimum number of data points in a given category in order to
        make a box and whisker. Otherwise, individual data points are
        plotted as in a strip plot.
    box_kwargs : dict, default None
        A dictionary of kwargs to be passed into `p.hbar()` or
        `p.vbar()` when constructing the boxes for the box plot.
    median_kwargs : dict, default None
        A dictionary of kwargs to be passed into `p.hbar()` or
        `p.vbar()` when constructing the median line for the box plot.
    whisker_kwargs : dict, default None
        A dictionary of kwargs to be passed into `p.segment()`
        when constructing the whiskers for the box plot.
    horizontal : bool or None, default None
        Deprecated. Use `q_axis`.
    val : hashable
        Deprecated, use `q`.
    kwargs
        Any kwargs to be passed to `bokeh.plotting.figure()` when
        instantiating the figure.

    Returns
    -------
    output : bokeh.plotting.Figure instance
        Plot populated with a strip plot.
    """
    # Protect against mutability of dicts
    box_kwargs = copy.copy(box_kwargs)
    median_kwargs = copy.copy(median_kwargs)
    whisker_kwargs = copy.copy(whisker_kwargs)
    jitter_kwargs = copy.copy(jitter_kwargs)
    marker_kwargs = copy.copy(marker_kwargs)
    parcoord_kwargs = copy.copy(parcoord_kwargs)

    # Set defaults
    if box_kwargs is None:
        box_kwargs = dict(line_color="gray", fill_alpha=0)
    if "color" not in box_kwargs and "line_color" not in box_kwargs:
        box_kwargs["line_color"] = "gray"
    if "fill_alpha" not in box_kwargs:
        box_kwargs["fill_alpha"] = 0

    if median_kwargs is None:
        median_kwargs = dict(line_color="gray")
    if "color" not in box_kwargs and "line_color" not in median_kwargs:
        median_kwargs["line_color"] = "gray"

    if whisker_kwargs is None:
        whisker_kwargs = dict(line_color="gray")
    if "color" not in box_kwargs and "line_color" not in whisker_kwargs:
        whisker_kwargs["line_color"] = "gray"

    if top_level == "box":
        p = strip(
            data=data,
            q=q,
            cats=cats,
            q_axis=q_axis,
            palette=palette,
            order=order,
            p=p,
            show_legend=show_legend,
            color_column=color_column,
            parcoord_column=parcoord_column,
            tooltips=tooltips,
            marker=marker,
            jitter=jitter,
            marker_kwargs=marker_kwargs,
            jitter_kwargs=jitter_kwargs,
            parcoord_kwargs=parcoord_kwargs,
            horizontal=horizontal,
            val=val,
            **kwargs,
        )

        p = box(
            data=data,
            q=q,
            cats=cats,
            q_axis=q_axis,
            palette=palette,
            order=order,
            p=p,
            display_points=False,
            whisker_caps=whisker_caps,
            min_data=min_data,
            box_kwargs=box_kwargs,
            median_kwargs=median_kwargs,
            whisker_kwargs=whisker_kwargs,
            horizontal=horizontal,
            val=val,
        )
    elif top_level == "strip":
        p = box(
            data=data,
            q=q,
            cats=cats,
            q_axis=q_axis,
            palette=palette,
            order=order,
            p=p,
            display_points=False,
            whisker_caps=whisker_caps,
            min_data=min_data,
            box_kwargs=box_kwargs,
            median_kwargs=median_kwargs,
            whisker_kwargs=whisker_kwargs,
            horizontal=horizontal,
            val=val,
            **kwargs,
        )

        p = strip(
            data=data,
            q=q,
            cats=cats,
            q_axis=q_axis,
            palette=palette,
            order=order,
            p=p,
            show_legend=show_legend,
            color_column=color_column,
            parcoord_column=parcoord_column,
            tooltips=tooltips,
            marker=marker,
            jitter=jitter,
            marker_kwargs=marker_kwargs,
            jitter_kwargs=jitter_kwargs,
            parcoord_kwargs=parcoord_kwargs,
            horizontal=horizontal,
            val=val,
        )
    else:
        raise RuntimeError("Invalid `top_level`. Allowed values are 'box' and 'strip'.")

    return p


def _get_cat_range(df, grouped, order, color_column, q_axis):
    if order is None:
        if isinstance(list(grouped.groups.keys())[0], tuple):
            factors = tuple(
                [tuple([str(k) for k in key]) for key in grouped.groups.keys()]
            )
        else:
            factors = tuple([str(key) for key in grouped.groups.keys()])
    else:
        if type(order[0]) in [list, tuple]:
            factors = tuple([tuple([str(k) for k in key]) for key in order])
        else:
            factors = tuple([str(entry) for entry in order])

    if q_axis == "x":
        cat_range = bokeh.models.FactorRange(*(factors[::-1]))
    elif q_axis == "y":
        cat_range = bokeh.models.FactorRange(*factors)

    if color_column is None:
        color_factors = factors
    else:
        color_factors = tuple(sorted(list(df[color_column].unique().astype(str))))

    return cat_range, factors, color_factors


def _cat_figure(df, grouped, q, order, color_column, q_axis, kwargs):
    cat_range, factors, color_factors = _get_cat_range(
        df, grouped, order, color_column, q_axis
    )

    kwargs = utils._fig_dimensions(kwargs)

    if q_axis == "x":
        if "x_axis_label" not in kwargs:
            kwargs["x_axis_label"] = q

        if "y_axis_type" in kwargs:
            warnings.warn("`y_axis_type` specified for categorical axis. Ignoring.")
            del kwargs["y_axis_type"]

        kwargs["y_range"] = cat_range
    elif q_axis == "y":
        if "y_axis_label" not in kwargs:
            kwargs["y_axis_label"] = q

        if "x_axis_type" in kwargs:
            warnings.warn("`x_axis_type` specified for categorical axis. Ignoring.")
            del kwargs["x_axis_type"]

        kwargs["x_range"] = cat_range

    return bokeh.plotting.figure(**kwargs), factors, color_factors


def _cat_source(df, cats, cols, color_column):
    cat_source, labels = utils._source_and_labels_from_cats(df, cats)

    if type(cols) in [list, tuple, pd.core.indexes.base.Index]:
        source_dict = {col: list(df[col].values) for col in cols}
    else:
        source_dict = {cols: list(df[cols].values)}

    source_dict["cat"] = cat_source
    if color_column in [None, "cat"]:
        source_dict["__label"] = labels
    else:
        source_dict["__label"] = list(df[color_column].astype(str).values)
        source_dict[color_column] = list(df[color_column].astype(str).values)

    return bokeh.models.ColumnDataSource(source_dict)


def _parcoord_source(data, q, cats, q_axis, parcoord_column, factors):
    if type(cats) not in [list, tuple]:
        cats = [cats]
        tuple_factors = False
    else:
        tuple_factors = True

    grouped_parcoord = data.groupby(parcoord_column)
    xs = []
    ys = []
    for t, g in grouped_parcoord:
        xy = []
        for _, r in g.iterrows():
            if tuple_factors:
                xy.append([tuple([r[cat] for cat in cats]), r[q]])
            else:
                xy.append([r[cats[0]], r[q]])

        if len(xy) > 1:
            xy.sort(key=lambda a: factors.index(a[0]))
            xs_pc = []
            ys_pc = []
            for pair in xy:
                xs_pc.append(pair[0])
                ys_pc.append(pair[1])

            if q_axis == "y":
                xs.append(xs_pc)
                ys.append(ys_pc)
            else:
                xs.append(ys_pc)
                ys.append(xs_pc)

    return bokeh.models.ColumnDataSource(dict(xs=xs, ys=ys))


def _outliers(data, min_data):
    if len(data) >= min_data:
        bottom, middle, top = np.percentile(data, [25, 50, 75])
        iqr = top - bottom
        outliers = data[(data > top + 1.5 * iqr) | (data < bottom - 1.5 * iqr)]
        return outliers
    else:
        return data


def _box_and_whisker(data, min_data):
    if len(data) >= min_data:
        middle = data.median()
        bottom = data.quantile(0.25)
        top = data.quantile(0.75)
        iqr = top - bottom
        top_whisker = max(data[data <= top + 1.5 * iqr].max(), top)
        bottom_whisker = min(data[data >= bottom - 1.5 * iqr].min(), bottom)
        return pd.Series(
            {
                "middle": middle,
                "bottom": bottom,
                "top": top,
                "top_whisker": top_whisker,
                "bottom_whisker": bottom_whisker,
            }
        )
    else:
        return pd.Series(
            {
                "middle": np.nan,
                "bottom": np.nan,
                "top": np.nan,
                "top_whisker": np.nan,
                "bottom_whisker": np.nan,
            }
        )


def _box_source(df, cats, q, cols, min_data):
    """Construct a data frame for making box plot."""
    # Need to reset index for use in slicing outliers
    df_source = df.reset_index(drop=True)

    if type(cats) in [list, tuple]:
        level = list(range(len(cats)))
    else:
        level = 0

    if cats is None:
        grouped = df_source
    else:
        grouped = df_source.groupby(cats, sort=False)

    # Data frame for boxes and whiskers
    df_box = grouped[q].apply(_box_and_whisker, min_data).unstack().reset_index()
    df_box = df_box.dropna()

    source_box = _cat_source(
        df_box, cats, ["middle", "bottom", "top", "top_whisker", "bottom_whisker"], None
    )

    # Data frame for outliers
    df_outliers = grouped[q].apply(_outliers, min_data)

    # If no cat has enough data, just use everything as an "outlier"
    if type(df_outliers) == pd.core.series.Series:
        df_outliers = df_source.copy()
    else:
        df_outliers = df_outliers.reset_index()

    df_outliers[cols] = df_source.loc[df_outliers.index, cols]

    source_outliers = _cat_source(df_outliers, cats, cols, None)

    return source_box, source_outliers

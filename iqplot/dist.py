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
from . import cat

try:
    import numba

    njit = numba.njit
except:
    njit = utils._dummy_jit


def ecdf(
    data=None,
    q=None,
    cats=None,
    q_axis="x",
    palette=None,
    order=None,
    p=None,
    show_legend=None,
    legend_label=None,
    legend_location="right",
    legend_orientation="vertical",
    legend_click_policy="hide",
    tooltips=None,
    complementary=False,
    kind="collection",
    style=None,
    arrangement="overlay",
    conf_int=False,
    ptiles=(2.5, 97.5),
    n_bs_reps=10000,
    marker="circle",
    marker_kwargs=None,
    line_kwargs=None,
    fill_kwargs=None,
    horizontal=None,
    val=None,
    click_policy=None,
    conf_int_kwargs=None,
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
    palette : list colors, or single color string
        If a list, color palette to use. If a single string representing
        a color, all glyphs are colored with that color. Default is
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
    legend_click_policy : str, default 'hide'
        Either 'hide', 'mute', or None; how the glyphs respond when the
        corresponding category is clicked in the legend.
    tooltips : list of 2-tuples
        Specification for tooltips as per Bokeh specifications. For
        example, if we want `col1` and `col2` tooltips, we can use
        `tooltips=[('label 1': '@col1'), ('label 2': '@col2')]`.
    complementary : bool, default False
        If True, plot the empirical complementary cumulative
        distribution function.
    kind : str, default 'collection'
        If 'collection', the figure is populated with a collection of
        ECDFs coded with colors based on the categorical variables. If
        'colored', the figure is populated with a single ECDF with
        circles colored based on the categorical variables.
    style : str, default 'staircase' for collection, 'dots' for colored
        The style of ECDF to make.

            - dots: Each data point is plotted as a dot.
            - staircase: ECDF is plotted as a traditional staircase.
            - formal: Strictly adhere to the definition of an ECDF.
    conf_int : bool, default False
        If True, display confidence interval of ECDF.
    ptiles : list, default (2.5, 97.5)
        The percentiles to use for the confidence interval. Ignored if
        `conf_int` is False.
    n_bs_reps : int, default 10,000
        Number of bootstrap replicates to do to compute confidence
        interval. Ignored if `conf_int` is False.
    marker : str, default 'circle'
        Name of marker to be used in the plot (ignored if `style` is
        'staircase'). Must be one of['asterisk', 'circle',
        'circle_cross', 'circle_x', 'cross', 'dash', 'diamond',
        'diamond_cross', 'hex', 'inverted_triangle', 'square',
        'square_cross', 'square_x', 'triangle', 'x']
    marker_kwargs : dict
        Keyword arguments to be passed to `p.circle()` or other relevant
        marker function.
    line_kwargs : dict
        Kwargs to be passed to `p.line()`, `p.ray()`, and `p.segment()`.
    fill_kwargs : dict
        Keyword arguments to pass to `p.patch()` when making the fill
        for the step-filled histogram or confidence intervals. Ignored
        if `style = 'step'` and `conf_int` is False. By default
        {"fill_alpha": 0.3, "line_alpha": 0}.
    horizontal : bool or None, default None
        Deprecated. Use `q_axis`.
    val : hashable
        Deprecated, use `q`.
    click_policy : str, default 'hide'
        Deprecated. Use `legend_click_policy`.
    conf_int_kwargs : dict
        Deprecated. Use `fill_kwargs`.
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
    fill_kwargs = copy.copy(fill_kwargs)

    # Check to make sure kind is ok
    if kind not in ['collection', 'colored']:
        raise RuntimeError("`kind` must be in `['collection', 'colored']")

    # Determine style
    if style is None:
        style = 'staircase' if kind =='collection' else 'dots'

    if conf_int:
        if type(ptiles) not in (list, tuple, np.ndarray) and len(ptiles) != 2:
            raise RuntimeError("`ptiles` must be a list or tuple of length 2.")
        else:
            ptiles = np.sort(ptiles)

    q, legend_click_policy, fill_kwargs = utils._parse_deprecations(
        q,
        q_axis,
        val,
        horizontal,
        "y",
        click_policy,
        legend_click_policy,
        conf_int_kwargs,
        fill_kwargs,
    )

    if style == "formal" and complementary:
        raise NotImplementedError("Complementary formal ECDFs not yet implemented.")

    if palette is None:
        palette = colorcet.b_glasbey_category10
    elif type(palette) == str:
        palette = [palette]

    if arrangement == "stack":
        if kind != "collection":
            raise RuntimeError("Must have kind='collection' if arrangment='stack'.")

        if show_legend is None:
            show_legend = False

        if show_legend:
            warnings.warn(
                "Cannot show legend with arrangement='stack'. There is no legend to show."
            )

        if cats is not None:
            return _stacked_ecdfs(
                data,
                q=q,
                cats=cats,
                q_axis=q_axis,
                palette=palette,
                order=order,
                tooltips=tooltips,
                complementary=complementary,
                kind=kind,
                style=style,
                conf_int=conf_int,
                ptiles=ptiles,
                n_bs_reps=n_bs_reps,
                marker=marker,
                marker_kwargs=marker_kwargs,
                line_kwargs=line_kwargs,
                fill_kwargs=fill_kwargs,
                **kwargs,
            )
    else:
        if show_legend is None:
            show_legend = True

    data, q, cats, show_legend = utils._data_cats(
        data, q, cats, show_legend, legend_label
    )
    order = utils._order_to_str(order)

    cats, cols = utils._check_cat_input(
        data, cats, q, None, None, tooltips, palette, order, marker_kwargs
    )

    kwargs = utils._fig_dimensions(kwargs)

    non_q_axis = "y" if q_axis == "x" else "x"
    if conf_int and f"{non_q_axis}_axis_type" in kwargs and kwargs[f"{non_q_axis}_axis_type"] == "log":
        warnings.warn(
            f"Cannot reliably draw confidence intervals with a {non_q_axis}-axis on a log scale because zero cannot be represented. Omitting confidence interval."
        )
        conf_int = False
    if (
        conf_int
        and f"{q_axis}_axis_type" in kwargs
        and kwargs[f"{q_axis}_axis_type"] == "log"
        and (data[q] <= 0).any()
    ):
        warnings.warn(
            f"Cannot draw confidence intervals with a {q_axis}-axis on a log scale because some values are negative. Any negative values will be omitted from the ECDF."
        )
        conf_int = False

    if marker_kwargs is None:
        marker_kwargs = {}
    if line_kwargs is None:
        line_kwargs = {}

    # Change any kwarg of "color" to line_color and fill_color
    marker_kwargs = utils._specific_fill_and_color_kwargs(marker_kwargs, "marker")
    line_kwargs = utils._specific_fill_and_color_kwargs(line_kwargs, "line")

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

    if fill_kwargs is None:
        fill_kwargs = {}
    if "fill_alpha" not in fill_kwargs:
        fill_kwargs["fill_alpha"] = 0.3
    if "line_alpha" not in fill_kwargs and "line_color" not in fill_kwargs:
        fill_kwargs["line_alpha"] = 0

    df = data.copy()

    if kind == "collection":
        if style == "dots" or tooltips is not None:
            df[y] = df.groupby(cats)[q].transform(_ecdf_y, complementary=complementary)
    elif kind == "colored":
        df[y] = df[q].transform(_ecdf_y, complementary=complementary)
        cols += [y]

    _, df["__label"] = utils._source_and_labels_from_cats(df, cats)
    cols += ["__label"]

    df = _sort_df(df, cats, order)

    if p is None:
        p = bokeh.plotting.figure(**kwargs)

    if style == "dots":
        marker_fun = utils._get_marker(p, marker)

    if tooltips is not None:
        p.add_tools(bokeh.models.HoverTool(tooltips=tooltips, name="hover_glyphs"))

    fill_fill_color_supplied = "fill_color" in fill_kwargs
    marker_fill_color_supplied = "fill_color" in marker_kwargs
    marker_line_color_supplied = "line_color" in marker_kwargs
    line_line_color_supplied = "line_color" in line_kwargs
    fill_fill_color_supplied = "fill_color" in fill_kwargs

    markers = []
    lines = []
    circles_high = []
    circles_low = []
    rays_high = []
    rays_low = []
    patches = []
    labels = []
    invisible_markers = []

    if kind == "collection":
        # Explicitly loop to enable click policies on the legend
        # (not possible with factors)
        for i, (name, g) in enumerate(df.groupby(cats, sort=False)):
            labels.append(g["__label"].iloc[0])
            if conf_int:
                if not fill_fill_color_supplied:
                    fill_kwargs["fill_color"] = palette[i % len(palette)]
                p, patch = _ecdf_conf_int(
                    p,
                    g[q],
                    complementary=complementary,
                    q_axis=q_axis,
                    n_bs_reps=n_bs_reps,
                    ptiles=ptiles,
                    **fill_kwargs,
                )
                patches.append(patch)

            if not marker_line_color_supplied:
                marker_kwargs["line_color"] = palette[i % len(palette)]
            if not marker_fill_color_supplied:
                marker_kwargs["fill_color"] = palette[i % len(palette)]
            if not line_line_color_supplied:
                line_kwargs["line_color"] = palette[i % len(palette)]

            if style == "staircase":
                p, new_line, new_ray_high, new_ray_low = _staircase_ecdf(
                    p,
                    data=g[q],
                    complementary=complementary,
                    q_axis=q_axis,
                    line_kwargs=line_kwargs,
                )
                lines.append(new_line)
                rays_high.append(new_ray_high)
                rays_low.append(new_ray_low)

            if style == "dots":
                if "name" not in marker_kwargs and tooltips is not None:
                    marker_kwargs["name"] = "hover_glyphs"

                if q_axis == "y":
                    markers.append(marker_fun(source=g, x=y, y=q, **marker_kwargs))
                else:
                    markers.append(marker_fun(source=g, x=q, y=y, **marker_kwargs))

            if style == "formal":
                (
                    p,
                    circle,
                    segment,
                    new_ray_high,
                    new_ray_low,
                    new_circle_high,
                    new_circle_low,
                ) = _formal_ecdf(
                    p,
                    data=g[q],
                    complementary=complementary,
                    q_axis=q_axis,
                    marker_kwargs=marker_kwargs,
                    line_kwargs=line_kwargs,
                )
                markers.append(circle)
                lines.append(segment)
                rays_high.append(new_ray_high)
                rays_low.append(new_ray_low)
                circles_high.append(new_circle_high)
                circles_low.append(new_circle_low)

            # Add transparent dots for hovering
            if style != "dots" and tooltips is not None:
                if q_axis == "y":
                    invisible_markers.append(
                        p.circle(
                            source=g,
                            x=y,
                            y=q,
                            name="hover_glyphs",
                            fill_alpha=0,
                            line_alpha=0,
                            size=7,
                        )
                    )
                else:
                    invisible_markers.append(
                        p.circle(
                            source=g,
                            x=q,
                            y=y,
                            name="hover_glyphs",
                            fill_alpha=0,
                            line_alpha=0,
                            size=7,
                        )
                    )
    elif kind == "colored":
        if style in ["formal", "staircase"]:
            raise RuntimeError(
                "Cannot have a formal or staircase ECDF with `kind='colored'`."
            )

        if conf_int:
            if "fill_color" not in fill_kwargs:
                fill_kwargs["fill_color"] = "gray"

            p, patch = _ecdf_conf_int(
                p,
                df[q],
                complementary=complementary,
                q_axis=q_axis,
                n_bs_reps=n_bs_reps,
                ptiles=ptiles,
                **fill_kwargs,
            )

        y = "__ECCDF" if complementary else "__ECDF"

        # Explicitly loop to enable click policies on the legend (not possible with factors)
        for i, (name, g) in enumerate(df.groupby(cats, sort=False)):
            source = bokeh.models.ColumnDataSource(g[cols])
            mkwargs = marker_kwargs
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
        legend_click_policy,
        labels,
        markers,
        lines,
        patches,
        rays_high,
        rays_low,
        circles_high,
        circles_low,
        invisible_markers,
    )


def histogram(
    data=None,
    q=None,
    cats=None,
    palette=None,
    order=None,
    q_axis="x",
    p=None,
    rug=None,
    rug_height=None,
    show_legend=None,
    legend_label=None,
    legend_location="right",
    legend_orientation="vertical",
    legend_click_policy="hide",
    tooltips=None,
    bins="freedman-diaconis",
    density=False,
    style=None,
    arrangement=None,
    mirror=False,
    hist_height=0.75,
    conf_int=False,
    ptiles=(2.5, 97.5),
    n_bs_reps=10000,
    line_kwargs=None,
    fill_kwargs=None,
    rug_kwargs=None,
    conf_int_kwargs=None,
    horizontal=None,
    val=None,
    click_policy=None,
    kind=None,
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
    palette : list colors, or single color string
        If a list, color palette to use. If a single string representing
        a color, all glyphs are colored with that color. Default is
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
    rug : bool, default True
        If True, also include a rug plot. If, however, `bins` is 'exact'
        or 'integer', the `rug` kwarg is ignored.
    rug_height : float, default None
        Height of the rug plot as a fraction of the highest point in the
        histograms. For 'overlay' arrangement, default is 0.05. For
        'stacked' arrangement, default is 0.2.
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
    legend_click_policy : str, default 'hide'
        Either 'hide', 'mute', or None; how the glyphs respond when the
        corresponding category is clicked in the legend.
    tooltips : list of 2-tuples
        Specification for tooltips as per Bokeh specifications. For
        example, if we want `col1` and `col2` tooltips, we can use
        `tooltips=[('label 1': '@col1'), ('label 2': '@col2')]`. Ignored
        if `rug` is False.
    bins : int, array_like, or str, default 'freedman-diaconis'
        If int or array_like, setting for `bins` kwarg to be passed to
        `np.histogram()`. If 'exact', then each unique value in the
        data gets its own bin. If 'integer', then integer data is
        assumed and each integer gets its own bin. If 'sqrt', uses the
        square root rule to determine number of bins. If
        `freedman-diaconis`, uses the Freedman-Diaconis rule for number
        of bins.
    density : bool, default False
        If True, normalize the histograms. Otherwise, base the
        histograms on counts.
    style : None or one of ['step', 'step_filled']
        Default for overlayed histograms is 'step' and for stacked
        histograms 'step_filled'. The exception is when `cont_int` is
        True, in which case `style` must be 'step'.
    arrangement : 'stack' or 'overlay'
        Arrangement of histograms. If 'overlay', histograms are overlaid
        on the same plot. If 'stack', histograms are stacked one on top
        of the other. By default, if `cats` is None or there is only one
        category, `arrangement` is 'overlay', and if there is more than
        one category, `arrangement` is 'stack'.
    mirror : bool, default False
        If True, reflect the histogram through zero. Ignored if
        `arrangement == 'overlay'`.
    hist_height : float, default 0.75
        Maximal height of histogram or its confidence interval as a
        fraction of available height along categorical axis. Only active
        when `arrangement` is 'stack'.
    conf_int : bool, default False
        If True, display confidence interval of histogram.
    ptiles : list, default [2.5, 97.5]
        The percentiles to use for the confidence interval. Ignored if
        `conf_int` is False.
    n_bs_reps : int, default 10,000
        Number of bootstrap replicates to do to compute confidence
        interval. Ignored if `conf_int` is False.
    line_kwargs : dict
        Keyword arguments to pass to `p.line()` in constructing the
        histograms. By default, {"line_width": 2}.
    fill_kwargs : dict
        Keyword arguments to pass to `p.patch()` when making the fill
        for the step-filled histogram or confidence intervals. Ignored
        if `style = 'step'` and `conf_int` is False. By default
        {"fill_alpha": 0.3, "line_alpha": 0}.
    rug_kwargs : dict
        Keyword arguments to pass to `p.multi_line()` when making the
        rug plot.
    horizontal : bool or None, default None
        Deprecated. Use `q_axis`.
    val : hashable
        Deprecated, use `q`.
    click_policy : str, default 'hide'
        Deprecated. Use `legend_click_policy`.
    conf_int_kwargs : dict
        Deprecated. Use `fill_kwargs`.
    kind : str, default 'step_filled'
        Deprecated. Use `style`.
    kwargs
        Any kwargs to be passed to `bokeh.plotting.figure()` when making
        the plot.

    Returns
    -------
    output : Bokeh figure
        Figure populated with histograms.

    Notes
    -----
    .. Confidence intervals for the histogram are computed using
       nonparametric bootstrap as follows. The bins are established as
       per user input via the `bins` kwarg. These bins are fixed for all
       bootstrap replicates. Then, for each bootstrap sample drawn, the
       histogram is computed for the bins. The confidence interval is
       then computed from these bootstrap samples.
    """
    # Protect against mutability of dicts
    line_kwargs = copy.copy(line_kwargs)
    fill_kwargs = copy.copy(fill_kwargs)
    rug_kwargs = copy.copy(rug_kwargs)

    if conf_int:
        if type(ptiles) not in (list, tuple, np.ndarray) and len(ptiles) != 2:
            raise RuntimeError("`ptiles` must be a list or tuple of length 2.")
        else:
            ptiles = np.sort(ptiles)

    # Check the deprecation of `kind` kwarg (not in utils._parse_deprecations because
    # `kind` is a valid kwarg for ECDFs)
    if kind is not None:
        if style != kind:
            raise RuntimeError(
                "`kind` and `style` in disagreement. Use `style`; `kind` is deprecated."
            )
        warnings.warn(
            f"`kind` is deprecated. Use `style`. Using style='{style}'.",
            DeprecationWarning,
        )

    if arrangement is None:
        arrangement = "overlay" if cats is None else "stack"

    if style is None:
        if conf_int:
            style = "step"
        else:
            if arrangement == "stack":
                style = "step_filled"
            else:
                style = "step"

    if arrangement == "stack":
        if rug_height is None:
            rug_height = 0.2
    elif arrangement != "overlay":
        raise RuntimeError(
            "Only allowed values for `arrangement` are 'stack' and 'overlay'."
        )
    elif rug_height is None:
        rug_height = 0.05

    if style == "step_filled" and conf_int:
        raise RuntimeError(
            "`style` must be 'step' when confidence intervals are displayed."
        )

    if conf_int and style == "step_filled":
        raise RuntimeError(
            f"`style` must be 'step' when confidence intervals are included."
        )

    q, legend_click_policy, fill_kwargs = utils._parse_deprecations(
        q,
        q_axis,
        val,
        horizontal,
        "y",
        click_policy,
        legend_click_policy,
        conf_int_kwargs,
        fill_kwargs,
    )

    if type(bins) == str and bins in ["integer", "exact"]:
        if rug is None:
            rug = False

        if rug:
            warnings.warn("Rug plot not generated for integer or exact bins.")
            rug = False
    elif rug is None:
        rug = True

    if palette is None:
        palette = colorcet.b_glasbey_category10
    elif type(palette) == str:
        palette = [palette]

    df, q, cats, show_legend = utils._data_cats(
        data, q, cats, show_legend, legend_label
    )
    order = utils._order_to_str(order)

    if arrangement == "stack":
        if show_legend is None:
            show_legend = False

        if show_legend:
            warnings.warn(
                "Cannot show legend with arrangement='stack'. There is no legend to show."
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

    # Defaults for histogram
    if line_kwargs is None:
        line_kwargs = {"line_width": 2}
    if fill_kwargs is None:
        fill_kwargs = {}
    if "fill_alpha" not in fill_kwargs:
        fill_kwargs["fill_alpha"] = 0.3
    if "line_alpha" not in fill_kwargs:
        fill_kwargs["line_alpha"] = 0

    # Defaults for rug_kwargs
    if rug_kwargs is None:
        rug_kwargs = dict(line_alpha=0.5, line_width=0.5)
    elif type(rug_kwargs) != dict:
        raise RuntimeError("`rug_kwargs` must be a dictionary.")
    if "alpha" not in rug_kwargs and "line_alpha" not in rug_kwargs:
        rug_kwargs["line_alpha"] = 0.5
    if "line_width" not in rug_kwargs:
        rug_kwargs["line_width"] = 0.5
    if "name" not in rug_kwargs:
        rug_kwargs["name"] = "hover_glyphs"

    # Change any kwarg of "color" to line_color and fill_color, same with alpha
    fill_kwargs = utils._specific_fill_and_color_kwargs(fill_kwargs, "fill")
    line_kwargs = utils._specific_fill_and_color_kwargs(line_kwargs, "line")
    rug_kwargs = utils._specific_fill_and_color_kwargs(rug_kwargs, "line")

    _, df["__label"] = utils._source_and_labels_from_cats(df, cats)
    cols += ["__label"]

    df = _sort_df(df, cats, order)

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

    if arrangement == "stack":
        if cats is not None:
            grouped = df.groupby(cats, sort=False)
            return _stacked_histograms(
                df,
                grouped,
                q,
                bins,
                density,
                palette,
                q_axis,
                order,
                p,
                mirror,
                hist_height,
                style,
                conf_int,
                ptiles,
                n_bs_reps,
                rug,
                rug_height,
                tooltips,
                line_kwargs,
                fill_kwargs,
                rug_kwargs,
                kwargs,
            )

    if p is None:
        kwargs = utils._fig_dimensions(kwargs)

        if "x_axis_label" not in kwargs:
            if q_axis == "y":
                if density:
                    kwargs["x_axis_label"] = "density"
                else:
                    kwargs["x_axis_label"] = "count"
            else:
                kwargs["x_axis_label"] = q

        if "y_axis_label" not in kwargs:
            if q_axis == "y":
                kwargs["y_axis_label"] = q
            else:
                if density:
                    kwargs["y_axis_label"] = "density"
                else:
                    kwargs["y_axis_label"] = "count"

        if q_axis == "y":
            if "x_range" not in kwargs:
                kwargs["x_range"] = bokeh.models.DataRange1d(start=0)
        else:
            if "y_range" not in kwargs:
                kwargs["y_range"] = bokeh.models.DataRange1d(start=0)

        p = bokeh.plotting.figure(**kwargs)

    # Explicitly loop to enable click policies on the legend (not possible with factors)
    max_height = 0
    lines = []
    labels = []
    patches = []
    for i, (name, g) in enumerate(df.groupby(cats, sort=False)):
        numerical_bins, e, f = _compute_histogram(g[q], bins, density)
        e0, f0 = _hist_for_plotting(e, f)

        if conf_int:
            fill_kwargs["fill_color"] = palette[i % len(palette)]

            q_vals = g[q].values
            f_reps = np.empty((n_bs_reps, len(f)))
            for j in range(n_bs_reps):
                _, _, f_reps[j, :] = _compute_histogram(
                    np.random.choice(q_vals, len(q_vals)), numerical_bins, density
                )

            f_ptiles = np.percentile(f_reps, ptiles, axis=0)

            _, f0_low = _hist_for_plotting(e, f_ptiles[0, :])
            _, f0_high = _hist_for_plotting(e, f_ptiles[1, :])
            if q_axis == "y":
                p, patch = utils._fill_between(
                    p, f0_low, e0, f0_high, e0, **fill_kwargs
                )
            else:
                p, patch = utils._fill_between(
                    p, e0, f0_low, e0, f0_high, **fill_kwargs
                )
            patches.append(patch)

        max_height = max(f0.max(), max_height)

        line_kwargs["color"] = palette[i % len(palette)]

        if q_axis == "y":
            lines.append(p.line(f0, e0, **line_kwargs))
        else:
            lines.append(p.line(e0, f0, **line_kwargs))
        labels.append(g["__label"].iloc[0])

        if style == "step_filled":
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
        y = [0, max_height * rug_height]

        for i, (name, g) in enumerate(df.groupby(cats, sort=False)):
            xs = [[q_val, q_val] for q_val in g[q].values]
            ys = [y] * len(g)

            if q_axis == "y":
                xs, ys = ys, xs

            cds = bokeh.models.ColumnDataSource(g)
            cds.data["__xs"] = xs
            cds.data["__ys"] = ys

            if "color" not in rug_kwargs and "line_color" not in rug_kwargs:
                p.multi_line(
                    source=cds,
                    xs="__xs",
                    ys="__ys",
                    line_color=palette[i % len(palette)],
                    **rug_kwargs,
                )
            else:
                p.multi_line(source=cds, xs="__xs", ys="__ys", **rug_kwargs)

    if tooltips is not None:
        p.add_tools(bokeh.models.HoverTool(tooltips=tooltips, name="hover_glyphs"))

    return _dist_legend(
        p,
        show_legend,
        legend_location,
        legend_orientation,
        legend_click_policy,
        labels,
        [],
        lines,
        patches,
        [],
        [],
        [],
        [],
        [],
    )


def spike(
    data=None,
    q=None,
    cats=None,
    palette=None,
    order=None,
    q_axis="x",
    p=None,
    show_legend=None,
    legend_label=None,
    legend_location="right",
    legend_orientation="vertical",
    legend_click_policy="hide",
    fraction=False,
    style=None,
    arrangement=None,
    spike_height=0.75,
    conf_int=False,
    ptiles=(2.5, 97.5),
    n_bs_reps=10000,
    line_kwargs=None,
    marker_kwargs=None,
    horizontal=None,
    val=None,
    click_policy=None,
    density=None,
    **kwargs,
):
    """
    Make a spike plot.

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
    palette : list colors, or single color string
        If a list, color palette to use. If a single string representing
        a color, all glyphs are colored with that color. Default is
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
    legend_click_policy : str, default 'hide'
        Either 'hide', 'mute', or None; how the glyphs respond when the
        corresponding category is clicked in the legend.
    fraction : bool, default False
        If True, the spike height is given by the fraction of data
        points having the given value. Otherwise, the height of a spike
        is given by the count of data points having the given value.
    style : None or one of ['spike', 'spike-dot', 'dot']
        'spike' gives a traditional spike plot. 'spike-dot' additionally
        features dots on top of the spikes, similar in appearance to a
        lollipop plot. 'dot' has the dot at the top of the spike, but
        the spike is not shown. Default is 'spike-dot', unless
        `conf_int` is True and the number of categorial values is
        greater than one, in which case the default is 'dot' (and only
        'dot' is allowed) for confidence intervals to avoid clashes.
    arrangement : 'stack' or 'overlay', default 'stack'
        Arrangement of spike plots. If 'overlay', spikes are overlaid
        on the same plot. If 'stack', spikes are stacked one on top
        of the other.
    spike_height : float, default 0.75
        Maximal height of spike or its confidence interval as a
        fraction of available height along categorical axis. Only active
        when `arrangement` is 'stack'.
    conf_int : bool, default False
        If True, display confidence interval of the spikes.
    ptiles : list with two elements, default [2.5, 97.5]
        The percentiles to use for the confidence interval. Ignored if
        `conf_int` is False.
    n_bs_reps : int, default 10,000
        Number of bootstrap replicates to do to compute confidence
        interval. Ignored if `conf_int` is False.
    marker_kwargs : dict
        Keyword arguments to be passed to `p.circle()` for dots at the
        top of spikes.
    line_kwargs : dict
        Keyword arguments to pass to `p.segment()` in constructing the
        spikes and confidence intervals. By default, {"line_width": 2}.
    horizontal : bool or None, default None
        Deprecated. Use `q_axis`.
    val : hashable
        Deprecated, use `q`.
    click_policy : str, default 'hide'
        Deprecated. Use `legend_click_policy`.
    kwargs
        Any kwargs to be passed to `bokeh.plotting.figure()` when making
        the plot.

    Returns
    -------
    output : Bokeh figure
        Figure populated with histograms.

    Notes
    -----
    .. Confidence intervals for the histogram are computed using
       nonparametric bootstrap as follows. The bins are established as
       per user input via the `bins` kwarg. These bins are fixed for all
       bootstrap replicates. Then, for each bootstrap sample drawn, the
       histogram is computed for the bins. The confidence interval is
       then computed from these bootstrap samples.
    """
    # Protect against mutability of dicts
    line_kwargs = copy.copy(line_kwargs)
    marker_kwargs = copy.copy(marker_kwargs)

    if conf_int:
        if type(ptiles) not in (list, tuple, np.ndarray) and len(ptiles) != 2:
            raise RuntimeError("`ptiles` must be a list or tuple of length 2.")
        else:
            ptiles = np.sort(ptiles)

    one_cat = cats is None or len(data.groupby(cats)) == 1

    if arrangement is None:
        arrangement = "overlay" if one_cat else "stack"

    # Use fraction, not density
    if density is not None:
        raise RuntimeError("For spike plots, use `fraction`, not `density`.")

    if style is None:
        if conf_int or (arrangement == "overlay" and not one_cat):
            style = "dot"
        else:
            style = "spike-dot"

    if style not in ["dot", "spike", "spike-dot"]:
        raise RuntimeError(
            "Valid values for `style` kwarg are 'dot', 'spike', and 'spike-dot'."
        )

    if arrangement not in ["stack", "overlay"]:
        raise RuntimeError(
            "Only allowed values for `arrangement` are 'stack' and 'overlay'."
        )

    if "spike" in style and conf_int:
        raise RuntimeError(
            "`style` must be 'dot' when confidence intervals are displayed."
        )

    if arrangement == "overlay" and cats is not None and "dot" not in style:
        raise RuntimeError(
            "`style` must be 'dot' or 'spike-dot' for overlay arrangement with more than one category."
        )

    q, legend_click_policy, _ = utils._parse_deprecations(
        q,
        q_axis,
        val,
        horizontal,
        "y",
        click_policy,
        legend_click_policy,
        None,
        {},
    )

    # Can't have a q be 'count' in Pandas v. 2.x Just make it always illegal
    if q == 'count':
        raise RuntimeError(
            'Cannot make a spike plot with a quantitative variable named "count." '
            + 'Rename the "count" column and start again.'
        )

    if palette is None:
        palette = colorcet.b_glasbey_category10
    elif type(palette) == str:
        palette = [palette]

    df, q, cats, show_legend = utils._data_cats(
        data, q, cats, show_legend, legend_label
    )
    order = utils._order_to_str(order)

    if arrangement == "stack":
        if show_legend is None:
            show_legend = False

        if show_legend:
            warnings.warn(
                "Cannot show legend with arrangement='stack'. There is no legend to show."
            )

    if show_legend is None:
        if cats is None:
            show_legend = False
        else:
            show_legend = True

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

    # Defaults for spikes
    if line_kwargs is None:
        line_kwargs = {"line_width": 2}

    # Marker kqargs use Bokeh defaults
    if marker_kwargs is None:
        marker_kwargs = {}

    # Change any kwarg of "color" to line_color and fill_color, same with alpha
    marker_kwargs = utils._specific_fill_and_color_kwargs(marker_kwargs, "marker")
    line_kwargs = utils._specific_fill_and_color_kwargs(line_kwargs, "line")

    _, df["__label"] = utils._source_and_labels_from_cats(df, cats)
    cols += ["__label"]

    df = _sort_df(df, cats, order)

    if arrangement == "stack":
        if cats is not None:
            grouped = df.groupby(cats, sort=False)
            return _stacked_spikes(
                df,
                grouped,
                q,
                palette,
                q_axis,
                order,
                p,
                spike_height,
                fraction,
                style,
                conf_int,
                ptiles,
                n_bs_reps,
                marker_kwargs,
                line_kwargs,
                kwargs,
            )

    marker_fill_color_supplied = "fill_color" in marker_kwargs
    marker_line_color_supplied = "line_color" in marker_kwargs
    line_line_color_supplied = "line_color" in line_kwargs

    if p is None:
        kwargs = utils._fig_dimensions(kwargs)

        if "x_axis_label" not in kwargs:
            if q_axis == "y":
                if fraction:
                    kwargs["x_axis_label"] = "fraction"
                else:
                    kwargs["x_axis_label"] = "count"
            else:
                kwargs["x_axis_label"] = q

        if "y_axis_label" not in kwargs:
            if q_axis == "y":
                kwargs["y_axis_label"] = q
            else:
                if fraction:
                    kwargs["y_axis_label"] = "fraction"
                else:
                    kwargs["y_axis_label"] = "count"

        if q_axis == "y":
            if "x_range" not in kwargs:
                kwargs["x_range"] = bokeh.models.DataRange1d(start=0)
        else:
            if "y_range" not in kwargs:
                kwargs["y_range"] = bokeh.models.DataRange1d(start=0)

        p = bokeh.plotting.figure(**kwargs)

    # Explicitly loop enable click policies on the legend (not possible with factors)
    lines = []
    markers = []
    labels = []

    # Confidence intervals
    if conf_int:

        @njit
        def _counts(ar, vals, frac):
            output = np.zeros(len(vals))
            for a in ar:
                output[np.searchsorted(vals, a)] += 1.0

            if frac:
                return output / len(ar)
            else:
                return output

        for i, (name, g) in enumerate(df.groupby(cats, sort=False)):
            x = g[q].values
            x_unique = np.unique(x)

            bs_reps = [
                _counts(
                    np.random.choice(x, replace=True, size=len(x)), x_unique, fraction
                )
                for _ in range(n_bs_reps)
            ]

            conf_ints = np.percentile(bs_reps, ptiles, axis=0)

            df_conf_int = pd.DataFrame(
                dict(
                    q=x_unique, __conf_low=conf_ints[0, :], __conf_high=conf_ints[1, :]
                )
            )

            if not line_line_color_supplied:
                line_kwargs["color"] = palette[i % len(palette)]

            if q_axis == "y":
                lines.append(
                    p.segment(
                        x0="__conf_low",
                        x1="__conf_high",
                        y0=q,
                        y1=q,
                        source=df_conf_int,
                        **line_kwargs,
                    )
                )
            else:
                lines.append(
                    p.segment(
                        x0=q,
                        x1=q,
                        y0="__conf_low",
                        y1="__conf_high",
                        source=df_conf_int,
                        **line_kwargs,
                    )
                )
            labels.append(g["__label"].iloc[0])

    # Spikes
    if "spike" in style:
        for i, (name, g) in enumerate(df.groupby(cats, sort=False)):
            # Because of changes in how value_counts works, we have to be careful about
            # renaming columns and indexes.
            # See https://pandas.pydata.org/docs/dev/whatsnew/v2.0.0.html#value-counts-sets-the-resulting-name-to-count)
            df_count = g[q].value_counts().reset_index()
            if pd.__version__ >= '2.0.0':
                df_count = df_count.rename(columns={'count': '__count'})
            else:
                df_count = df_count.rename(columns={"index": q, q: "__count"})

            if fraction:
                df_count["__count"] /= df_count["__count"].sum()

            if not line_line_color_supplied:
                line_kwargs["color"] = palette[i % len(palette)]

            if q_axis == "y":
                lines.append(
                    p.segment(
                        x0=0, x1="__count", y0=q, y1=q, source=df_count, **line_kwargs
                    )
                )
            else:
                lines.append(
                    p.segment(
                        x0=q, x1=q, y0=0, y1="__count", source=df_count, **line_kwargs
                    )
                )
            labels.append(g["__label"].iloc[0])

    # Overlay dots
    if "dot" in style:
        for i, (name, g) in enumerate(df.groupby(cats, sort=False)):
            df_count = g[q].value_counts().reset_index()
            if pd.__version__ >= '2.0.0':
                df_count = df_count.rename(columns={'count': '__count'})
            else:
                df_count = df_count.rename(columns={"index": q, q: "__count"})

            if fraction:
                df_count["__count"] /= df_count["__count"].sum()

            if not marker_line_color_supplied:
                marker_kwargs["line_color"] = palette[i % len(palette)]
            if not marker_fill_color_supplied:
                marker_kwargs["fill_color"] = palette[i % len(palette)]

            if q_axis == "y":
                markers.append(
                    p.circle(x="__count", y=q, source=df_count, **marker_kwargs)
                )
            else:
                markers.append(
                    p.circle(x=q, y="__count", source=df_count, **marker_kwargs)
                )

            labels.append(g["__label"].iloc[0])

    return _dist_legend(
        p,
        show_legend,
        legend_location,
        legend_orientation,
        legend_click_policy,
        labels,
        markers,
        lines,
        [],
        [],
        [],
        [],
        [],
        [],
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
    p : bokeh.plotting.Figure instance
        Plot populated with ECDF.
    line : bokeh.models.glyph.LineGlyph.Line instance
        Line of staircase, used for constructing clickable legend
    ray_high : bokeh.models.glyph.LineGlyph.Ray instance
        Ray for top of ECDF, used for constructing clickable legend.
    ray_low : bokeh.models.glyph.LineGlyph.Ray instance
        Ray for bottom of ECDF, used for constructing clickable legend.
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
            ray_high = p.ray(x=1, y=x[0], length=0, angle=-np.pi / 2, **line_kwargs)
            ray_low = p.ray(x=0, y=x[-1], length=0, angle=np.pi / 2, **line_kwargs)
        else:
            ray_low = p.ray(x=0, y=x[0], length=0, angle=-np.pi / 2, **line_kwargs)
            ray_high = p.ray(x=1, y=x[-1], length=0, angle=np.pi / 2, **line_kwargs)
    elif q_axis == "x":
        if complementary:
            ray_high = p.ray(x=x[0], y=1, length=0, angle=np.pi, **line_kwargs)
            ray_low = p.ray(x=x[-1], y=0, length=0, angle=0, **line_kwargs)
        else:
            ray_low = p.ray(x=x[0], y=0, length=0, angle=np.pi, **line_kwargs)
            ray_high = p.ray(x=x[-1], y=1, length=0, angle=0, **line_kwargs)

    return p, line, ray_high, ray_low


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
    p : bokeh.plotting.Figure instance
        Plot populated with ECDF.
    segment : bokeh.models.glyph.LineGlyph.Segment instance
        Line of staircase, used for constructing clickable legend
    ray_high : bokeh.models.glyph.LineGlyph.Ray instance
        Ray for top of ECDF, used for constructing clickable legend.
    ray_low : bokeh.models.glyph.LineGlyph.Ray instance
        Ray for bottom of ECDF, used for constructing clickable legend.
    circle_high : bokeh.models.glyph.LineGlyph.Ray instance
        Open circle for top of ECDF, used for constructing clickable
        legend.
    circle_low : bokeh.models.glyph.LineGlyph.Ray instance
        Open circle for bottom of ECDF, used for constructing clickable
        legend.
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
        ray_low = p.ray(x=0, y=x[0], angle=-np.pi / 2, length=0, **line_kwargs)
        ray_high = p.ray(x=1, y=x[-1], angle=np.pi / 2, length=0, **line_kwargs)
        circle = p.circle(y, x, **marker_kwargs)
        circle_low = p.circle([0], [0], **unfilled_kwargs)
        circle_high = p.circle(y[:-1], x[1:], **unfilled_kwargs)
    elif q_axis == "x":
        segment = p.segment(x[:-1], y[:-1], x[1:], y[:-1], **line_kwargs)
        ray_low = p.ray(x=x[0], y=0, angle=np.pi, length=0, **line_kwargs)
        ray_high = p.ray(x=x[-1], y=1, angle=0, length=0, **line_kwargs)
        circle = p.circle(x, y, **marker_kwargs)
        circle_low = p.circle([0], [0], **unfilled_kwargs)
        circle_high = p.circle(x[1:], y[:-1], **unfilled_kwargs)

    return p, circle, segment, ray_high, ray_low, circle_high, circle_low


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


def _stacked_ecdfs(
    data,
    q=None,
    cats=None,
    q_axis="x",
    palette=None,
    order=None,
    tooltips=None,
    complementary=False,
    kind="collection",
    style="dots",
    conf_int=False,
    ptiles=(2.5, 97.5),
    n_bs_reps=10000,
    marker="circle",
    marker_kwargs=None,
    line_kwargs=None,
    fill_kwargs=None,
    **kwargs,
):
    ps = []

    if type(cats) in [list, tuple] and len(cats) == 1:
        cats = cats[0]

    # Protect against mutability and get copies
    df = data.copy()
    kwargs = copy.copy(kwargs)
    marker_kwargs = copy.copy(marker_kwargs)
    line_kwargs = copy.copy(line_kwargs)
    fill_kwargs = copy.copy(fill_kwargs)

    df = _sort_df(df, cats, order)

    if (
        "frame_width" not in kwargs
        and "width" not in kwargs
        and "plot_width" not in kwargs
    ):
        if q_axis == "y":
            kwargs["frame_width"] = 100
    if (
        "frame_height" not in kwargs
        and "height" not in kwargs
        and "plot_height" not in kwargs
    ):
        if q_axis == "x":
            kwargs["frame_height"] = 100
    if "min_border" not in kwargs:
        kwargs["min_border"] = kwargs.pop("min_border", 5)

    if marker_kwargs is None:
        marker_kwargs = {}
    if line_kwargs is None:
        line_kwargs = {}
    if fill_kwargs is None:
        fill_kwargs = {}

    marker_fill_color_supplied = "fill_color" in marker_kwargs
    marker_line_color_supplied = "line_color" in marker_kwargs
    line_line_color_supplied = "line_color" in line_kwargs
    fill_fill_color_supplied = "fill_color" in fill_kwargs

    title = kwargs.pop("title", None)
    if title is not None and q_axis == "y":
        raise RuntimeError(
            "`title` is not an allowed kwarg when q_axis is 'y' and `arrangment` is 'stack'."
        )

    for i, (name, g) in enumerate(df.groupby(cats)):
        color = palette[i % len(palette)]
        if not marker_fill_color_supplied:
            marker_kwargs["fill_color"] = color
        if not marker_line_color_supplied:
            marker_kwargs["line_color"] = color
        if not line_line_color_supplied:
            line_kwargs["line_color"] = color
        if not fill_fill_color_supplied:
            fill_kwargs["fill_color"] = color

        if q_axis == "x":
            kwargs["y_axis_label"] = str(name)
            if i == 0:
                kwargs["title"] = title
            else:
                kwargs["title"] = None
        else:
            kwargs["x_axis_label"] = q
            kwargs["title"] = str(name)

        ps.append(
            ecdf(
                data=g,
                q=q,
                q_axis=q_axis,
                tooltips=tooltips,
                complementary=complementary,
                kind=kind,
                style=style,
                conf_int=conf_int,
                ptiles=ptiles,
                n_bs_reps=n_bs_reps,
                marker=marker,
                marker_kwargs=marker_kwargs,
                line_kwargs=line_kwargs,
                fill_kwargs=fill_kwargs,
                **kwargs,
            )
        )

    if q_axis == "x":
        for i, _ in enumerate(ps[:-1]):
            ps[i].xaxis.visible = False
            ps[i].xaxis.axis_label = None
            ps[i].x_range = ps[-1].x_range
            ps[i].y_range = ps[-1].y_range

        for i, _ in enumerate(ps):
            ps[i].yaxis.minor_tick_out = 0
            ps[i].yaxis.axis_label_text_font_style = "bold"
            ps[i].yaxis.axis_label_text_color = "#696969"

        return bokeh.layouts.gridplot(ps, ncols=1)
    else:
        for i, _ in enumerate(ps[1:]):
            ps[i + 1].yaxis.visible = False
            ps[i + 1].yaxis.axis_label = None
            ps[i + 1].x_range = ps[0].x_range
            ps[i + 1].y_range = ps[0].y_range

        for i, _ in enumerate(ps):
            ps[i].xaxis.minor_tick_out = 0
            ps[i].xaxis.major_label_orientation = np.pi / 3
            ps[i].title.align = "center"
            ps[i].title.text_font_style = "bold"
            ps[i].title.text_color = "#696969"

        return bokeh.layouts.gridplot(ps, ncols=len(ps))


def _stacked_histograms(
    df,
    grouped,
    q,
    bins,
    density,
    palette,
    q_axis,
    order,
    p,
    mirror,
    hist_height,
    style,
    conf_int,
    ptiles,
    n_bs_reps,
    rug,
    rug_height,
    tooltips,
    line_kwargs,
    fill_kwargs,
    rug_kwargs,
    kwargs,
):
    # Protect against mutability and get copies
    line_kwargs = copy.copy(line_kwargs)
    fill_kwargs = copy.copy(fill_kwargs)

    line_line_color_supplied = "line_color" in line_kwargs
    fill_fill_color_supplied = "fill_color" in fill_kwargs

    if p is None:
        p, _, _ = cat._cat_figure(df, grouped, q, order, None, q_axis, kwargs)

    f0_max = 0.0
    plot_data = {}
    for i, (name, g) in enumerate(grouped):
        numerical_bins, e, f = _compute_histogram(g[q].values, bins, density)
        e0, f0 = _hist_for_plotting(e, f)

        plot_data[name] = dict(e0=e0, f0=f0)

        # Record f0_max
        f0_max = max(f0_max, f0.max())

        if conf_int:
            q_vals = g[q].values
            f_reps = np.empty((n_bs_reps, len(f)))
            for j in range(n_bs_reps):
                _, _, f_reps[j, :] = _compute_histogram(
                    np.random.choice(q_vals, len(q_vals)), numerical_bins, density
                )

            f_ptiles = np.percentile(f_reps, ptiles, axis=0)

            _, f0_low = _hist_for_plotting(e, f_ptiles[0, :])
            _, f0_high = _hist_for_plotting(e, f_ptiles[1, :])

            # Store the plot data
            plot_data[name]["f0_low"] = f0_low
            plot_data[name]["f0_high"] = f0_high

            # Record max f0
            f0_max = max(f0_max, f0_high.max())

    if not density:
        scale = 1.0 / f0_max * hist_height / 2

    for i, (name, plot_data_dict) in enumerate(plot_data.items()):
        if not fill_fill_color_supplied:
            fill_kwargs["fill_color"] = palette[i % len(palette)]
        if not line_line_color_supplied:
            line_kwargs["line_color"] = palette[i % len(palette)]

        if density:
            if conf_int:
                scale = 1.0 / plot_data_dict["f0_high"].max() * hist_height / 2
            else:
                scale = 1.0 / plot_data_dict["f0"].max() * hist_height / 2

        if conf_int:
            f0_low_cat = [
                (*name, f0_val) if type(name) == tuple else (name, f0_val)
                for f0_val in scale * plot_data_dict["f0_low"]
            ]
            f0_high_cat = [
                (*name, f0_val) if type(name) == tuple else (name, f0_val)
                for f0_val in scale * plot_data_dict["f0_high"]
            ]

            if q_axis == "y":
                p, patch = utils._fill_between(
                    p,
                    f0_low_cat,
                    plot_data_dict["e0"],
                    f0_high_cat,
                    plot_data_dict["e0"],
                    **fill_kwargs,
                )
            else:
                p, patch = utils._fill_between(
                    p,
                    plot_data_dict["e0"],
                    f0_low_cat,
                    plot_data_dict["e0"],
                    f0_high_cat,
                    **fill_kwargs,
                )

            if mirror:
                f0_low_cat = [
                    (*name, f0_val) if type(name) == tuple else (name, f0_val)
                    for f0_val in -scale * plot_data_dict["f0_low"]
                ]
                f0_high_cat = [
                    (*name, f0_val) if type(name) == tuple else (name, f0_val)
                    for f0_val in -scale * plot_data_dict["f0_high"]
                ]
                if q_axis == "y":
                    p, patch = utils._fill_between(
                        p,
                        f0_low_cat,
                        plot_data_dict["e0"],
                        f0_high_cat,
                        plot_data_dict["e0"],
                        **fill_kwargs,
                    )
                else:
                    p, patch = utils._fill_between(
                        p,
                        plot_data_dict["e0"],
                        f0_low_cat,
                        plot_data_dict["e0"],
                        f0_high_cat,
                        **fill_kwargs,
                    )

        # y-values for histogram, appropriately scaled
        f0 = plot_data_dict["f0"] * scale
        f0_cat = [
            (*name, f0_val) if type(name) == tuple else (name, f0_val) for f0_val in f0
        ]

        if mirror:
            f0_cat += list(
                reversed(
                    [
                        (*name, -f0_val) if type(name) == tuple else (name, -f0_val)
                        for f0_val in f0
                    ]
                )
            )
            e0 = np.concatenate((plot_data_dict["e0"], plot_data_dict["e0"][::-1]))
        else:
            e0 = plot_data_dict["e0"]

        # Line of histogram
        if q_axis == "y":
            p.line(f0_cat, e0, **line_kwargs)
            if style == "step_filled":
                p.patch(f0_cat, e0, **fill_kwargs)
        else:
            p.line(e0, f0_cat, **line_kwargs)
            if style == "step_filled":
                p.patch(e0, f0_cat, **fill_kwargs)

        # Add rug
        if rug:
            for i, (name, g) in enumerate(grouped):
                xs = [[x, x] for x in g[q]]
                y0_cat = [
                    (*name, rug_height * hist_height / 2)
                    if type(name) == tuple
                    else (name, rug_height * hist_height / 2)
                    for _ in range(len(g))
                ]

                if mirror:
                    ys = [(y0, y0[:-1] + (-y0[-1],)) for y0 in y0_cat]
                else:
                    ys = [(y0, y0[:-1] + (0,)) for y0 in y0_cat]

                cds = bokeh.models.ColumnDataSource(g)
                cds.data["__xs"] = xs
                cds.data["__ys"] = ys

                if "color" not in rug_kwargs and "line_color" not in rug_kwargs:
                    p.multi_line(
                        source=cds,
                        xs="__xs",
                        ys="__ys",
                        line_color=palette[i % len(palette)],
                        **rug_kwargs,
                    )
                else:
                    p.multi_line(source=cds, xs="__xs", ys="__ys", **rug_kwargs)

    if rug and tooltips is not None:
        p.add_tools(bokeh.models.HoverTool(tooltips=tooltips, name="hover_glyphs"))

    return p


def _stacked_spikes(
    df,
    grouped,
    q,
    palette,
    q_axis,
    order,
    p,
    spike_height,
    fraction,
    style,
    conf_int,
    ptiles,
    n_bs_reps,
    marker_kwargs,
    line_kwargs,
    kwargs,
):
    # Protect against mutability and get copies
    line_kwargs = copy.copy(line_kwargs)
    marker_kwargs = copy.copy(marker_kwargs)

    line_line_color_supplied = "line_color" in line_kwargs
    marker_fill_color_supplied = "fill_color" in marker_kwargs
    marker_line_color_supplied = "line_color" in marker_kwargs

    if p is None:
        p, _, _ = cat._cat_figure(df, grouped, q, order, None, q_axis, kwargs)

    # Compute confidence intervals, keeping track of maximum possible spike height
    if conf_int:

        @njit
        def _counts(ar, vals, frac):
            output = np.zeros(len(vals))
            for a in ar:
                output[np.searchsorted(vals, a)] += 1.0

            if frac:
                return output / len(ar)
            else:
                return output

        conf_ints_dict = dict()
        max_spike = 0
        for i, (name, g) in enumerate(grouped):
            x = g[q].values
            x_unique = np.unique(x)

            bs_reps = [
                _counts(
                    np.random.choice(x, replace=True, size=len(x)), x_unique, fraction
                )
                for _ in range(n_bs_reps)
            ]

            conf_ints = np.percentile(bs_reps, ptiles, axis=0)

            if not fraction:
                max_spike = max(max_spike, conf_ints.max())

            conf_ints_dict[name] = pd.DataFrame(
                {
                    q: x_unique,
                    "__conf_low": conf_ints[0, :],
                    "__conf_high": conf_ints[1, :],
                }
            )
    elif not fraction:
        counts = grouped[q].value_counts().rename("__count").reset_index()
        max_spike = counts["__count"].max()

    if not fraction:
        scale = 1.0 / max_spike * spike_height / 2

    for i, (name, g) in enumerate(grouped):
        # Confidence intervals
        if conf_int:
            if fraction:
                scale = (
                    1.0 / conf_ints_dict[name]["__conf_high"].max() * spike_height / 2
                )

            conf_ints_dict[name]["__conf_cat_low"] = [
                (*name, val) if type(name) == tuple else (name, val)
                for val in scale * conf_ints_dict[name]["__conf_low"]
            ]

            conf_ints_dict[name]["__conf_cat_high"] = [
                (*name, val) if type(name) == tuple else (name, val)
                for val in scale * conf_ints_dict[name]["__conf_high"]
            ]

            if not line_line_color_supplied:
                line_kwargs["color"] = palette[i % len(palette)]

            if q_axis == "y":
                p.segment(
                    x0="__conf_cat_low",
                    x1="__conf_cat_high",
                    y0=q,
                    y1=q,
                    source=conf_ints_dict[name],
                    **line_kwargs,
                )
            else:
                p.segment(
                    x0=q,
                    x1=q,
                    y0="__conf_cat_low",
                    y1="__conf_cat_high",
                    source=conf_ints_dict[name],
                    **line_kwargs,
                )

        # Make a count data frame for spikes and dots
        df_count = g[q].value_counts().reset_index()
        if pd.__version__ >= '2.0.0':
            df_count = df_count.rename(columns={'count': '__count'})
        else:
            df_count = df_count.rename(columns={"index": q, q: "__count"})

        if fraction:
            df_count["__count"] /= df_count["__count"].sum()

        # For now, enforce fraction
        if fraction:
            df_count["__count"] /= df_count["__count"].sum()

        # Scaling to fit properly with counting
        if not conf_int and fraction:
            scale = 1.0 / np.max(df_count["__count"]) * spike_height / 2

        # Compute counts with the categorical value included
        df_count["__count_cat"] = [
            (*name, val) if type(name) == tuple else (name, val)
            for val in scale * df_count["__count"]
        ]

        # Spikes
        if "spike" in style:
            df_count["__count_cat_base"] = [
                (*name, 0) if type(name) == tuple else (name, 0)
                for _ in df_count["__count"]
            ]

            if not line_line_color_supplied:
                line_kwargs["color"] = palette[i % len(palette)]

            if q_axis == "y":
                p.segment(
                    x0="__count_cat_base",
                    x1="__count_cat",
                    y0=q,
                    y1=q,
                    source=df_count,
                    **line_kwargs,
                )
            else:
                p.segment(
                    x0=q,
                    x1=q,
                    y0="__count_cat_base",
                    y1="__count_cat",
                    source=df_count,
                    **line_kwargs,
                )

        # Overlay dots
        if "dot" in style:
            if not line_line_color_supplied:
                line_kwargs["color"] = palette[i % len(palette)]

            if not marker_line_color_supplied:
                marker_kwargs["line_color"] = palette[i % len(palette)]
            if not marker_fill_color_supplied:
                marker_kwargs["fill_color"] = palette[i % len(palette)]

            if q_axis == "y":
                p.circle(x="__count_cat", y=q, source=df_count, **marker_kwargs)
            else:
                p.circle(x=q, y="__count_cat", source=df_count, **marker_kwargs)

    return p


def _ecdf_conf_int(
    p,
    data,
    complementary=False,
    q_axis="x",
    n_bs_reps=10000,
    ptiles=(2.5, 97.5),
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
    rays_high,
    rays_low,
    circles_high,
    circles_low,
    invisible_markers,
):
    """Add a legend to a histogram, spike, or ECDF plot."""
    if show_legend:
        if len(markers) > 0:
            if len(lines) > 0:
                if len(patches) > 0:
                    if len(invisible_markers) > 0:
                        items = [
                            (
                                label,
                                [
                                    line,
                                    patch,
                                    ray_high,
                                    ray_low,
                                    circle_high,
                                    circle_low,
                                    invisible_marker,
                                    marker,
                                ],
                            )
                            for label, line, patch, ray_high, ray_low, circle_high, circle_low, invisible_marker, marker in zip(
                                labels,
                                lines,
                                patches,
                                rays_high,
                                rays_low,
                                circles_high,
                                circles_low,
                                invisible_markers,
                                markers,
                            )
                        ]
                    else:
                        items = [
                            (
                                label,
                                [
                                    line,
                                    patch,
                                    ray_high,
                                    ray_low,
                                    circle_high,
                                    circle_low,
                                    marker,
                                ],
                            )
                            for label, line, patch, ray_high, ray_low, circle_high, circle_low, marker in zip(
                                labels,
                                lines,
                                patches,
                                rays_high,
                                rays_low,
                                circles_high,
                                circles_low,
                                markers,
                            )
                        ]
                else:
                    if len(invisible_markers) > 0:
                        items = [
                            (
                                label,
                                [
                                    line,
                                    ray_high,
                                    ray_low,
                                    circle_high,
                                    circle_low,
                                    invisible_marker,
                                    marker,
                                ],
                            )
                            for label, line, ray_high, ray_low, circle_high, circle_low, invisible_marker, marker in zip(
                                labels,
                                lines,
                                rays_high,
                                rays_low,
                                circles_high,
                                circles_low,
                                invisible_markers,
                                markers,
                            )
                        ]
                    elif not rays_high:
                        items = [
                            (
                                label,
                                [
                                    line,
                                    marker,
                                ],
                            )
                            for label, line, marker in zip(
                                labels,
                                lines,
                                markers,
                            )
                        ]
                    else:
                        items = [
                            (
                                label,
                                [
                                    line,
                                    ray_high,
                                    ray_low,
                                    circle_high,
                                    circle_low,
                                    marker,
                                ],
                            )
                            for label, line, ray_high, ray_low, circle_high, circle_low, marker in zip(
                                labels,
                                lines,
                                rays_high,
                                rays_low,
                                circles_high,
                                circles_low,
                                markers,
                            )
                        ]
            else:
                if len(patches) > 0:
                    if len(invisible_markers) > 0:
                        items = [
                            (label, [marker, invisible_marker, patch])
                            for label, marker, invisible_marker, patch in zip(
                                labels, markers, invisible_markers, patches
                            )
                        ]
                    else:
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
                if len(rays_high) > 0:
                    if len(invisible_markers) > 0:
                        items = [
                            (label, [line, patch, ray_high, ray_low, invisible_marker])
                            for label, line, patch, ray_high, ray_low, invisible_marker in zip(
                                labels,
                                lines,
                                patches,
                                rays_high,
                                rays_low,
                                invisible_markers,
                            )
                        ]
                    else:
                        items = [
                            (label, [line, patch, ray_high, ray_low])
                            for label, line, patch, ray_high, ray_low in zip(
                                labels, lines, patches, rays_high, rays_low
                            )
                        ]
                else:
                    if len(invisible_markers) > 0:
                        items = [
                            (label, [line, patch, invisible_marker])
                            for label, line, patch, invisible_marker in zip(
                                labels, lines, patches, invisible_markers
                            )
                        ]
                    else:
                        items = [
                            (label, [line, patch])
                            for label, line, patch in zip(labels, lines, patches)
                        ]
            else:
                if len(rays_high) > 0:
                    if len(invisible_markers) > 0:
                        items = [
                            (label, [line, ray_high, ray_low, invisible_marker])
                            for label, line, ray_high, ray_low, invisible_marker in zip(
                                labels, lines, rays_high, rays_low, invisible_markers
                            )
                        ]
                    else:
                        items = [
                            (label, [line, ray_high, ray_low])
                            for label, line, ray_high, ray_low in zip(
                                labels, lines, rays_high, rays_low
                            )
                        ]
                else:
                    if len(invisible_markers) > 0:
                        items = [
                            (label, [line, invisible_marker])
                            for label, line, invisible_marker in zip(
                                labels, lines, invisible_markers
                            )
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


def _sort_df(df, cats, order):
    if order is not None and cats is not None:
        if type(cats) in [list, tuple]:
            df["__sort"] = df.apply(lambda r: order.index(tuple(r[cats])), axis=1)
        else:
            df["__sort"] = df.apply(lambda r: order.index(r[cats]), axis=1)
        df = df.sort_values(by="__sort")

    return df


def _compute_histogram(data, bins, density):
    """Computes the bins and edges of a histogram."""
    if type(bins) == str and bins == "sqrt":
        bins = int(np.ceil(np.sqrt(len(data))))
    elif type(bins) == str and bins == "freedman-diaconis":
        h = 2 * (np.percentile(data, 75) - np.percentile(data, 25)) / np.cbrt(len(data))
        if h == 0.0:
            bins = 3
        else:
            bins = int(np.ceil((data.max() - data.min()) / h))

    f, e = np.histogram(data, bins=bins, density=density)

    return bins, e, f


def _hist_for_plotting(e, f):
    """Takes output e and f from _compute_histogram(), and generates
    x, y values for plotting the histogram."""
    e0 = np.empty(2 * len(e))
    f0 = np.empty(2 * len(e))
    e0[::2] = e
    e0[1::2] = e
    f0[0] = 0
    f0[-1] = 0
    f0[1:-1:2] = f
    f0[2:-1:2] = f

    return e0, f0

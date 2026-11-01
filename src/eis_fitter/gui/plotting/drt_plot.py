"""
DRT plot renderer — reimplements plot_drt_results for embedded canvas.
Renders onto a provided matplotlib Figure (no plt.show/plt.close).
"""
import numpy as np
from matplotlib.gridspec import GridSpec
from eis_fitter.gui.theme import ThemeManager


def _style_axes(ax, p):
    """Apply theme palette colors to an axes object."""
    ax.set_facecolor(p['plot_axes_bg'])
    ax.tick_params(colors=p['plot_tick'])
    ax.xaxis.label.set_color(p['plot_text'])
    ax.yaxis.label.set_color(p['plot_text'])
    ax.title.set_color(p['plot_text'])
    for spine in ax.spines.values():
        spine.set_color(p['plot_tick'])


def render_drt_results(fig, freq, zreal, zimag, drt_result, sign_conv,
                       data_file=None, palette=None):
    """
    Render 6-panel DRT figure onto the provided Figure.

    Layout:
      Top row:    empty (left) + Settings table (right)
      Middle row: DRT spectrum (left) + Nyquist (right)
      Bottom row: Bode (left) + Residuals (right)
    """
    from eis_fitter.gui.engine_bridge import get_engine
    engine = get_engine()

    p = palette or ThemeManager.instance().current_palette

    fig.set_facecolor(p['plot_bg'])

    gamma = drt_result['gamma']
    log_tau = drt_result['log_tau']
    Z_fit = drt_result['Z_fit']
    R_inf = drt_result['R_inf']
    R_pol = drt_result['R_pol']
    lam = drt_result['lambda_used']
    rmse = drt_result['rmse']
    r2 = drt_result['r2']

    omega = 2.0 * np.pi * freq
    tau_grid = 10.0 ** log_tau

    import os
    gs = GridSpec(3, 2, figure=fig,
                  height_ratios=[0.95, 1.5, 0.85],
                  hspace=0.75, wspace=0.40,
                  left=0.07, right=0.91, bottom=0.08, top=0.94)

    # -- Top-left: Title --
    ax_title = fig.add_subplot(gs[0, 0])
    ax_title.set_axis_off()
    _title_lines = "Distribution of Relaxation Times\nDRT Analysis"
    if data_file:
        _title_lines += f"\n{os.path.basename(data_file)}"
    ax_title.text(0.05, 0.55, _title_lines,
                  transform=ax_title.transAxes,
                  fontsize=13, fontweight='bold',
                  ha='left', va='center', color=p['plot_text'])

    # -- Top-right: Settings table --
    ax_tbl = fig.add_subplot(gs[0, 1])
    ax_tbl.set_axis_off()

    lam_method = drt_result.get('lambda_method', 'N/A')
    reg_order = drt_result.get('reg_order', 2)
    data_use = drt_result.get('data_use', 'combined')
    inductance = drt_result.get('inductance', 'N/A')
    fwhm_coeff = drt_result.get('fwhm_coeff', 0.5)
    non_neg = drt_result.get('non_negative', True)
    L_ind = drt_result.get('L_ind', 0.0)

    ord_label = f"{reg_order}{'st' if reg_order == 1 else 'nd'} derivative"
    lam_display = f"{lam_method}"
    if lam_method == 'custom':
        lam_display += f" (\u03bb = {lam:.2e})"
    else:
        lam_display += f" \u2192 \u03bb = {lam:.2e}"

    tbl_data = [
        ['Model', 'DRT (Tikhonov + Gaussian RBF)'],
        ['\u03bb selection', lam_display],
        ['Regularisation', ord_label],
        ['Data use', data_use],
        ['Inductance', inductance],
        ['FWHM coeff.', f'{fwhm_coeff}'],
        ['Non-negativity', 'Yes' if non_neg else 'No'],
        ['R\u221e', f'{R_inf:.6f} \u03a9'],
        ['R_pol', f'{R_pol:.6f} \u03a9'],
        ['L_ind', f'{L_ind:.4e} H'],
        ['RMSE', f'{rmse:.6f} \u03a9'],
        ['R\u00b2', f'{r2:.6f}'],
    ]

    tbl = ax_tbl.table(cellText=tbl_data, cellLoc='center', loc='center',
                       colWidths=[0.35, 0.62], bbox=[0, 0, 1, 1])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.0, 1.25)
    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor(p['table_border'])
        if row % 2 == 0:
            cell.set_facecolor(p['table_row_even'])
        else:
            cell.set_facecolor(p['table_row_odd'])
        cell.set_text_props(color=p['plot_text'])
        if col == 0:
            cell.set_text_props(fontweight='bold', fontsize=8,
                                color=p['plot_text'])

    # -- Middle-left: DRT spectrum --
    ax1 = fig.add_subplot(gs[1, 0])
    _style_axes(ax1, p)
    ax1.fill_between(log_tau, 0, gamma, color='steelblue', alpha=0.3)
    ax1.plot(log_tau, gamma, '-', lw=2, color='navy',
             label=r'$\gamma(\log\tau)$')
    ax1.set_xlabel(r'$\log_{10}(\tau\,/\,\mathrm{s})$', fontsize=12)
    ax1.set_ylabel(r'$\gamma\;(\Omega)$', fontsize=12)
    ax1.set_title("DRT Spectrum", fontsize=12, fontweight='bold')
    ax1.set_xlim(log_tau[0], log_tau[-1])
    ax1.set_ylim(bottom=0,
                 top=gamma.max() * 1.35 if gamma.max() > 0 else 1)
    ax1.grid(True, alpha=0.3, ls='--', color=p['plot_grid'])

    # Secondary x-axis: frequency
    ax1_top = ax1.twiny()
    ax1_top.set_xlim(ax1.get_xlim())
    tick_positions = np.linspace(log_tau[0], log_tau[-1], 5)
    tick_freqs = 1.0 / (2.0 * np.pi * 10.0 ** tick_positions)
    ax1_top.set_xticks(tick_positions)
    ax1_top.set_xticklabels([engine.eng_str(f, 'Hz') for f in tick_freqs],
                             fontsize=7)
    ax1_top.set_xlabel("Frequency", fontsize=9, color=p['text_muted'])
    ax1_top.tick_params(colors=p['plot_tick'])

    # Mark peaks
    from scipy.signal import find_peaks as _find_peaks
    peak_indices, _ = _find_peaks(
        gamma, height=0.05 * gamma.max(), distance=5,
        prominence=0.02 * gamma.max())
    import matplotlib.pyplot as plt
    peak_colors_drt = plt.cm.tab10(
        np.linspace(0, 0.9, max(len(peak_indices), 1)))
    box_bg = p['plot_axes_bg'] if p['name'] == 'dark' else 'white'
    for ip, pi in enumerate(peak_indices):
        f_peak = 1.0 / (2.0 * np.pi * tau_grid[pi])
        ax1.annotate(
            f"f={engine.eng_str(f_peak, 'Hz')}\n"
            f"\u03c4={tau_grid[pi]:.2e} s\n"
            f"\u03b3={gamma[pi]:.4f} \u03a9",
            (log_tau[pi], gamma[pi]),
            fontsize=7, fontweight='bold', color=peak_colors_drt[ip],
            textcoords='offset points', xytext=(0, 12), ha='center',
            arrowprops=dict(arrowstyle='->', color=peak_colors_drt[ip],
                            lw=1.2),
            bbox=dict(boxstyle='round,pad=0.2', facecolor=box_bg, alpha=0.8))

    info_box_face = p['plot_axes_bg'] if p['name'] == 'dark' else 'wheat'
    info_txt = (f"R\u221e = {R_inf:.4f} \u03a9\n"
                f"R_pol = {R_pol:.4f} \u03a9\n"
                f"\u03bb = {lam:.2e}\n"
                f"RMSE = {rmse:.4e}\n"
                f"R\u00b2 = {r2:.6f}\n"
                f"Peaks: {len(peak_indices)}")
    ax1.text(0.98, 0.98, info_txt, transform=ax1.transAxes, fontsize=7,
             ha='right', va='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor=info_box_face, alpha=0.85),
             color=p['plot_text'])
    ax1.legend(fontsize=9, loc='upper left')

    # -- Middle-right: Nyquist --
    ax2 = fig.add_subplot(gs[1, 1])
    _style_axes(ax2, p)
    y_meas = engine._nyquist_sign(zimag, sign_conv)
    y_fit = engine._nyquist_sign(Z_fit.imag, sign_conv)

    ax2.plot(zreal, y_meas, 'o', ms=6, mfc='none', mec='gray',
             mew=1.5, label='Measured', zorder=2, alpha=0.7)
    ax2.plot(Z_fit.real, y_fit, '-', lw=2.5, color='crimson',
             label='DRT Fit', zorder=5)

    ylabel_nq = "-Z'' (\u03a9)" if sign_conv == "negative" else "Z'' (\u03a9)"
    ax2.set_xlabel("Z' (\u03a9)", fontsize=11)
    ax2.set_ylabel(ylabel_nq, fontsize=11)
    ax2.set_title("Nyquist Plot (DRT Fit)", fontsize=11, fontweight='bold')
    ax2.set_aspect('equal', adjustable='box')
    ax2.grid(True, alpha=0.3, ls='--', color=p['plot_grid'])
    ax2.legend(fontsize=9)
    engine.apply_eng_format(ax2, 'both')

    x_pad = (zreal.max() - zreal.min()) * 0.05 + 1e-9
    y_pad = (y_meas.max() - y_meas.min()) * 0.05 + 1e-9
    ax2.set_xlim(zreal.min() - x_pad, zreal.max() + x_pad)
    ax2.set_ylim(y_meas.min() - y_pad, y_meas.max() + y_pad)
    engine._annotate_freq_on_nyquist(
        ax2, freq, zreal, y_meas, step=max(1, len(freq) // 4))

    # -- Bottom-left: Bode --
    ax3 = fig.add_subplot(gs[2, 0])
    _style_axes(ax3, p)
    Zmod_m = np.sqrt(zreal**2 + zimag**2)
    Zph_m = np.degrees(np.arctan2(zimag, zreal))
    Zmod_f = np.abs(Z_fit)
    Zph_f = np.degrees(np.arctan2(Z_fit.imag, Z_fit.real))

    ax3.set_xlabel("Frequency (Hz)", fontsize=11)
    ax3.set_xscale('log')
    ax3.set_ylabel("|Z| (\u03a9)", fontsize=11, color='tab:blue')
    ax3.loglog(freq, Zmod_m, 'o', ms=4, color='tab:blue',
               label='|Z| meas', alpha=0.6)
    ax3.loglog(freq, Zmod_f, '-', lw=2.5, color='navy',
               label='|Z| DRT fit')
    ax3.tick_params(axis='y', labelcolor='tab:blue')

    ax3b = ax3.twinx()
    ax3b.set_ylabel("Phase (\u00b0)", fontsize=11, color='tab:red')
    ax3b.semilogx(freq, Zph_m, 's', ms=3, color='tab:red',
                  label='Phase meas', alpha=0.6)
    ax3b.semilogx(freq, Zph_f, '-', lw=2.5, color='darkred',
                  label='Phase DRT')
    ax3b.tick_params(axis='y', labelcolor='tab:red')

    ax3.set_title("Bode Plot (DRT Fit)", fontsize=11, fontweight='bold')
    ax3.grid(True, alpha=0.3, ls='--', color=p['plot_grid'])
    h1, l1 = ax3.get_legend_handles_labels()
    h2, l2 = ax3b.get_legend_handles_labels()
    ax3.legend(h1 + h2, l1 + l2, loc='best', fontsize=8)
    engine.apply_eng_format(ax3, 'x')
    engine.apply_eng_format(ax3, 'y')

    # -- Bottom-right: Residuals --
    ax4 = fig.add_subplot(gs[2, 1])
    _style_axes(ax4, p)
    res_real = zreal - Z_fit.real
    res_imag = zimag - Z_fit.imag
    Z_mag = np.sqrt(zreal**2 + zimag**2)
    Z_mag_safe = np.where(Z_mag > 1e-15, Z_mag, 1e-15)
    res_real_pct = 100.0 * res_real / Z_mag_safe
    res_imag_pct = 100.0 * res_imag / Z_mag_safe

    ax4.semilogx(freq, res_real_pct, 'o-', ms=3, color='steelblue',
                 label="Z' residual %")
    ax4.semilogx(freq, res_imag_pct, 's-', ms=3, color='tomato',
                 label="Z'' residual %")
    ax4.axhline(0, color=p['plot_tick'], lw=0.8, ls='--')
    ax4.axhline(1.0, color=p['text_muted'], lw=0.6, ls=':', alpha=0.5)
    ax4.axhline(-1.0, color=p['text_muted'], lw=0.6, ls=':', alpha=0.5)
    ax4.set_xlabel("Frequency (Hz)", fontsize=11)
    ax4.set_ylabel("Relative Residual (%)", fontsize=11)
    ax4.set_title("DRT Fit Residuals", fontsize=11, fontweight='bold')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3, ls='--', color=p['plot_grid'])
    engine.apply_eng_format(ax4, 'x')

    # GridSpec defines spacing for the tables and secondary axes.

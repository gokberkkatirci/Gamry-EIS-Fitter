"""
ECM fit plot renderer — reimplements plot_fit_with_components for embedded canvas.
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


def render_ecm_fit(fig, freq, zreal, zimag, Z_fit, Z_base, voigt_list,
                   params, n_voigt, metrics, sign_conv,
                   kk_result=None, data_file=None, palette=None):
    """
    Render 6-panel ECM fit figure onto the provided Figure.

    Layout:
      Top row:    Settings table (left) + ECM circuit diagram (right)
      Middle row: Nyquist (left) + Bode (right)
      Bottom row: Circuit residuals (left) + KK residuals (right)
    """
    from eis_fitter.gui.engine_bridge import get_engine
    engine = get_engine()

    p = palette or ThemeManager.instance().current_palette

    fig.set_facecolor(p['plot_bg'])

    omega = 2.0 * np.pi * freq
    colors = __import__('matplotlib').cm.tab10(
        np.linspace(0, 0.7, max(n_voigt, 1)))
    has_kk = kk_result is not None

    import os
    gs = GridSpec(3, 2, figure=fig,
                  height_ratios=[0.85, 1.5, 0.85],
                  hspace=0.65, wspace=0.38,
                  left=0.07, right=0.91, bottom=0.08, top=0.88)

    # -- Suptitle --
    _title_file = ''
    if data_file:
        _title_file = f'\n{os.path.basename(data_file)}'
    fig.suptitle(
        f"ECM Fit Results \u2014 L + Rs + {n_voigt}\u00d7(R\u2016CPE)"
        f"{_title_file}",
        fontsize=14, fontweight='bold', x=0.50, ha='center', va='top',
        color=p['plot_text'])

    # -- Top-left: Settings table --
    ax_tbl = fig.add_subplot(gs[0, 0])
    ax_tbl.set_axis_off()

    tbl_data = [
        ['Model', f'L + Rs + {n_voigt}\u00d7(R\u2016CPE)'],
        ['Voigt elements', f'{n_voigt}'],
        ['Parameters', f'{metrics["n_params"]}'],
        ['Data points', f'{len(freq)}'],
        ['Freq. range', f'{engine.eng_str(freq.min(), "Hz")} \u2013 '
                        f'{engine.eng_str(freq.max(), "Hz")}'],
        ['RMSE', f'{metrics["rmse"]:.6f} \u03a9'],
        ['R\u00b2', f'{metrics["r2"]:.6f}'],
    ]
    if 'bic' in metrics:
        tbl_data.append(['BIC', f'{metrics["bic"]:.2f}'])
    if has_kk:
        kk_status = 'PASS' if kk_result['passed'] else 'FAIL'
        tbl_data.append(['KK status',
                         f'{kk_status}  (M={kk_result["n_rc"]}, '
                         f'res={kk_result["mean_res_pct"]:.3f}%)'])

    tbl = ax_tbl.table(cellText=tbl_data, colLabels=['Setting', 'Value'],
                       cellLoc='center', colLoc='center', loc='center',
                       colWidths=[0.38, 0.58], bbox=[0, 0, 1, 1])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.0, 1.35)
    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor(p['table_border'])
        if row == 0:
            cell.set_facecolor(p['table_header_bg'])
            cell.set_text_props(color=p['table_header_fg'],
                                fontweight='bold', fontsize=9)
        elif row % 2 == 0:
            cell.set_facecolor(p['table_row_even'])
            cell.set_text_props(color=p['plot_text'])
        else:
            cell.set_facecolor(p['table_row_odd'])
            cell.set_text_props(color=p['plot_text'])
        if col == 0 and row > 0:
            cell.set_text_props(fontweight='bold', fontsize=8,
                                color=p['plot_text'])

    # -- Top-right: ECM circuit diagram --
    ax_circ = fig.add_subplot(gs[0, 1])
    engine._draw_ecm_circuit(ax_circ, n_voigt, params)

    # -- Middle-left: Nyquist --
    ax1 = fig.add_subplot(gs[1, 0])
    _style_axes(ax1, p)
    y_meas = engine._nyquist_sign(zimag, sign_conv)
    y_fit = engine._nyquist_sign(Z_fit.imag, sign_conv)

    ax1.plot(zreal, y_meas, 'o', ms=6, mfc='none', mec='gray',
             mew=1.5, label='Measured', zorder=2, alpha=0.7)
    ax1.plot(Z_fit.real, y_fit, '-', lw=3, color=p['plot_text'],
             label='Circuit Fit', zorder=5)

    if has_kk:
        y_kk = engine._nyquist_sign(kk_result['Z_kk'].imag, sign_conv)
        ax1.plot(kk_result['Z_kk'].real, y_kk, '--', lw=2, color='magenta',
                 label=f"KK Fit (M={kk_result['n_rc']})", zorder=4, alpha=0.8)

    # Individual arcs with characteristic-frequency offset
    omega_fine = np.logspace(
        np.log10(omega.min()), np.log10(omega.max()), 500)
    zreal_lo, zreal_hi = zreal.min(), zreal.max()

    arc_data = []
    for i in range(n_voigt):
        idx = 2 + 3 * i
        Ri, Qi, ai = params[idx], params[idx+1], params[idx+2]
        rq_i = Ri * Qi
        tau_i = (rq_i ** (1.0 / ai)) if (rq_i > 0 and ai > 0) else rq_i
        Zv_fine = engine.Z_voigt(omega_fine, Ri, Qi, ai)
        arc_data.append((Ri, Qi, ai, tau_i, Zv_fine))

    Rs = params[1]
    for i in range(n_voigt):
        Ri, Qi, ai, tau_i, Zv_fine_i = arc_data[i]
        omega_char_i = (1.0 / tau_i) if tau_i > 1e-30 else omega_fine[0]
        real_offset = Rs
        for j in range(i):
            Rj, Qj, aj, _, _ = arc_data[j]
            Zj_at_char = engine.Z_voigt(np.array([omega_char_i]), Rj, Qj, aj)
            real_offset += Zj_at_char.real[0]
        x_arc = real_offset + Zv_fine_i.real
        y_arc = engine._nyquist_sign(Zv_fine_i.imag, sign_conv)
        arc_mask = (x_arc >= zreal_lo) & (x_arc <= zreal_hi)
        xm, ym = x_arc[arc_mask], y_arc[arc_mask]
        if len(xm) > 2:
            ax1.fill_between(xm, 0, ym, color=colors[i], alpha=0.10, zorder=1)
        ax1.plot(xm, ym, '--', lw=1.5, color=colors[i],
                 label=f"RC{i+1} (R={Ri:.4f}, a={ai:.3f})",
                 zorder=3, alpha=0.7)

    engine._annotate_freq_on_nyquist(
        ax1, freq, zreal, y_meas, step=max(1, len(freq) // 4))

    ylabel_nq = "-Z'' (\u03a9)" if sign_conv == "negative" else "Z'' (\u03a9)"
    ax1.set_xlabel("Z' (\u03a9)", fontsize=11)
    ax1.set_ylabel(ylabel_nq, fontsize=11)
    ax1.set_title(f"Nyquist - L+Rs+{n_voigt}x(R||CPE)",
                  fontsize=11, fontweight='bold')
    ax1.set_aspect('equal', adjustable='datalim')
    ax1.grid(True, alpha=0.3, ls='--', color=p['plot_grid'])
    ax1.legend(fontsize=7, loc='best')
    engine.apply_eng_format(ax1, 'both')

    x_pad = (zreal.max() - zreal.min()) * 0.05 + 1e-9
    y_pad = (y_meas.max() - y_meas.min()) * 0.05 + 1e-9
    ax1.set_xlim(zreal.min() - x_pad, zreal.max() + x_pad)
    ax1.set_ylim(y_meas.min() - y_pad, y_meas.max() + y_pad)

    # Detailed parameters are in the circuit diagram and text export.

    # -- Middle-right: Bode --
    ax2 = fig.add_subplot(gs[1, 1])
    _style_axes(ax2, p)
    Zmod_m = np.sqrt(zreal**2 + zimag**2)
    Zph_m = np.degrees(np.arctan2(zimag, zreal))
    Zmod_f = np.abs(Z_fit)
    Zph_f = np.degrees(np.arctan2(Z_fit.imag, Z_fit.real))

    ax2.set_xlabel("Frequency (Hz)", fontsize=11)
    ax2.set_xscale('log')
    ax2.set_ylabel("|Z| (\u03a9)", fontsize=11, color='tab:blue')
    ax2.loglog(freq, Zmod_m, 'o', ms=4, color='tab:blue',
               label='|Z| meas', alpha=0.6)
    ax2.loglog(freq, Zmod_f, '-', lw=2.5, color='navy',
               label='|Z| circuit fit')
    ax2.tick_params(axis='y', labelcolor='tab:blue')

    if has_kk:
        Zmod_kk = np.abs(kk_result['Z_kk'])
        ax2.loglog(freq, Zmod_kk, '--', lw=1.5, color='magenta',
                   label=f"|Z| KK (M={kk_result['n_rc']})", alpha=0.8)

    ax2b = ax2.twinx()
    ax2b.set_ylabel("Phase (\u00b0)", fontsize=11, color='tab:red')
    ax2b.semilogx(freq, Zph_m, 's', ms=3, color='tab:red',
                  label='Phase meas', alpha=0.6)
    ax2b.semilogx(freq, Zph_f, '-', lw=2.5, color='darkred',
                  label='Phase circuit')
    if has_kk:
        Zph_kk = np.degrees(np.arctan2(kk_result['Z_kk'].imag,
                                        kk_result['Z_kk'].real))
        ax2b.semilogx(freq, Zph_kk, '--', lw=1.5, color='orchid',
                       label='Phase KK', alpha=0.8)
    ax2b.tick_params(axis='y', labelcolor='tab:red')

    ax2.set_title("Bode Plot", fontsize=11, fontweight='bold')
    ax2.grid(True, alpha=0.3, ls='--', color=p['plot_grid'])
    h1, l1 = ax2.get_legend_handles_labels()
    h2, l2 = ax2b.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, loc='best', fontsize=7)
    engine.apply_eng_format(ax2, 'x')
    engine.apply_eng_format(ax2, 'y')

    # -- Bottom-left: Circuit residuals --
    ax3 = fig.add_subplot(gs[2, 0])
    _style_axes(ax3, p)
    res_real = zreal - Z_fit.real
    res_imag = zimag - Z_fit.imag
    ax3.semilogx(freq, res_real, 'o-', ms=3, color='steelblue',
                 label='Zreal res.')
    ax3.semilogx(freq, res_imag, 's-', ms=3, color='tomato',
                 label='Zimag res.')
    ax3.axhline(0, color=p['plot_tick'], lw=0.8, ls='--')
    ax3.set_xlabel("Frequency (Hz)", fontsize=11)
    ax3.set_ylabel("Residual (\u03a9)", fontsize=11)
    ax3.set_title("Circuit Fit Residuals", fontsize=11, fontweight='bold')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3, ls='--', color=p['plot_grid'])
    engine.apply_eng_format(ax3, 'x')
    engine.apply_eng_format(ax3, 'y')

    # -- Bottom-right: KK residuals --
    ax4 = fig.add_subplot(gs[2, 1])
    _style_axes(ax4, p)
    if has_kk:
        ax4.semilogx(freq, kk_result['res_real_pct'], 'o-', ms=3,
                      color='steelblue', label='KK Zreal res. %')
        ax4.semilogx(freq, kk_result['res_imag_pct'], 's-', ms=3,
                      color='tomato', label='KK Zimag res. %')
        ax4.axhline(0, color=p['plot_tick'], lw=0.8, ls='--')
        ax4.axhline(1.0, color=p['text_muted'], lw=0.6, ls=':', alpha=0.5)
        ax4.axhline(-1.0, color=p['text_muted'], lw=0.6, ls=':', alpha=0.5)
        status_txt = 'PASS' if kk_result['passed'] else 'FAIL'
        status_clr = p['primary'] if kk_result['passed'] else p['danger']
        box_bg = p['plot_axes_bg']
        ax4.text(0.98, 0.98,
                 f"KK: {status_txt}\nM={kk_result['n_rc']}\n"
                 f"mean |res|={kk_result['mean_res_pct']:.3f}%",
                 transform=ax4.transAxes, fontsize=7,
                 ha='right', va='top', color=status_clr, fontweight='bold',
                 bbox=dict(boxstyle='round', facecolor=box_bg, alpha=0.8))
    else:
        ax4.text(0.5, 0.5, "No KK data", transform=ax4.transAxes,
                 ha='center', va='center', fontsize=12, color=p['text_muted'])
    ax4.set_xlabel("Frequency (Hz)", fontsize=11)
    ax4.set_ylabel("Relative Residual (%)", fontsize=11)
    ax4.set_title("KK Residuals", fontsize=11, fontweight='bold')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3, ls='--', color=p['plot_grid'])
    engine.apply_eng_format(ax4, 'x')

    # GridSpec defines spacing for the tables and secondary axes.

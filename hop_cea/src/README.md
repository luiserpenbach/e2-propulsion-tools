# h2cea – thrust chamber reference analyses (Handbook H2-PRP-HBK-001 Rev B)

Thermochemistry and first-order front end for RESA. All baseline inputs live in `h2cea/config.py`.

    pip install rocketcea CoolProp numpy scipy matplotlib
    python run_all.py          # ~25 s, writes CSV/PNG/SVG + results.json into out/

Outputs: 01 O/F sweep (liquid vs gaseous N2O card), 02 operating line vs architecture Rev A,
03 chamber species, 04 oxidiser enthalpy sensitivity, 05 area ratio / separation,
06 CEA transport vs textbook approximations, 07 Bartz profiles on E2-REG-1,
08 jacket pressure regime map, 09 Hall–Mudawar CHF estimate, 10 finite-area combustor,
decomposition temperatures, acoustics, 11 jacket outlet states and p–h path.
`notion_figs.py` builds the compact SVGs used in the Notion handbook.

For RESA: take gas properties from `cea_si.point()`, operating points from
`operating_line.state()`, and the jacket inlet state from `n2o.jacket_pressures()` + tank enthalpy.

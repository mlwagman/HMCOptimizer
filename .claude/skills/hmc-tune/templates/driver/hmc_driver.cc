// QCD HMC driver for HMCOptimizer.
//
// Single-source, three-binary build (integrator chosen at compile time via
// -DHMC_INTEGRATOR=LeapFrog|MinimumNorm2|ForceGradient). Every other knob
// is a runtime CLI flag so one binary can serve an entire tuning sweep.
//
// Action content (2+1 flavour Wilson-Clover, Symanzik-improved gauge,
// stout-smeared fermions, Hasenbusch-preconditioned light sector,
// rational-approximated strange sector) follows the pattern in
// Grid-TXQCD/production/gen_qcd_cfgs.cc but uses only stock Grid classes.
//
// Build:
//   make -f Makefile.hmc_driver INTEGRATOR=MinimumNorm2
//
// Invoke (example):
//   srun -n N hmc_driver_omelyan \
//       --grid 16.16.16.32 --mpi 1.1.1.4 \
//       --beta 6.1 --u0 0.8326 \
//       --m-light -0.2450 --m-strange -0.2450 --csw 1.24931 \
//       --stout-rho 0.125 --stout-nsmear 1 \
//       --md-steps 10 --traj-length 1.41421356 \
//       --cg-tol 1e-8 --cg-max 30000 \
//       --hasenbusch 0.0145,0.045,0.108,0.25 --hasenbusch-top 1.0 \
//       --rat-lo 1e-4 --rat-hi 200.0 --rat-degree 16 \
//       --n-trajectories 20 --no-metropolis-until 0 \
//       --starting-type ColdStart --start-trajectory 0 \
//       --ckpt-prefix ./cfgs/ckpoint --ckpt-interval 5
//
// Flags not recognised here are parsed by Grid_init (e.g. --threads, --dslash-generic).
//
// Integrator macro defaults to MinimumNorm2 if not provided.
#ifndef HMC_INTEGRATOR
#define HMC_INTEGRATOR MinimumNorm2
#endif
#define HMC_STRINGIFY_(x) #x
#define HMC_STRINGIFY(x) HMC_STRINGIFY_(x)

#include <Grid/Grid.h>
#include <cstdlib>
#include <sstream>
#include <string>
#include <vector>

using namespace Grid;

namespace {

// --- CLI helpers ----------------------------------------------------------

bool has_flag(int argc, char **argv, const std::string &f) {
  return GridCmdOptionExists(argv, argv + argc, f);
}

std::string arg_str(int argc, char **argv, const std::string &f,
                    const std::string &fallback) {
  if (!has_flag(argc, argv, f)) return fallback;
  return GridCmdOptionPayload(argv, argv + argc, f);
}

double arg_double(int argc, char **argv, const std::string &f, double fallback) {
  if (!has_flag(argc, argv, f)) return fallback;
  return std::stod(GridCmdOptionPayload(argv, argv + argc, f));
}

int arg_int(int argc, char **argv, const std::string &f, int fallback) {
  if (!has_flag(argc, argv, f)) return fallback;
  return std::stoi(GridCmdOptionPayload(argv, argv + argc, f));
}

std::vector<double> arg_double_csv(int argc, char **argv,
                                   const std::string &f) {
  std::vector<double> out;
  if (!has_flag(argc, argv, f)) return out;
  std::string payload = GridCmdOptionPayload(argv, argv + argc, f);
  if (payload.empty()) return out;
  std::stringstream ss(payload);
  std::string item;
  while (std::getline(ss, item, ',')) {
    if (!item.empty()) out.push_back(std::stod(item));
  }
  return out;
}

}  // namespace

int main(int argc, char **argv) {
  Grid_init(&argc, &argv);

  // --- Runtime parameters -------------------------------------------------
  const double beta        = arg_double(argc, argv, "--beta",        6.1);
  const double u0          = arg_double(argc, argv, "--u0",          0.8326053);
  const double m_light     = arg_double(argc, argv, "--m-light",    -0.2450);
  const double m_strange   = arg_double(argc, argv, "--m-strange",  -0.2450);
  const double csw         = arg_double(argc, argv, "--csw",         1.0);
  const double stout_rho   = arg_double(argc, argv, "--stout-rho",   0.125);
  const int    stout_nsmr  = arg_int   (argc, argv, "--stout-nsmear",1);
  const int    md_steps    = arg_int   (argc, argv, "--md-steps",    10);
  const double traj_len    = arg_double(argc, argv, "--traj-length", std::sqrt(2.0));
  const double cg_tol      = arg_double(argc, argv, "--cg-tol",      1.0e-8);
  const int    cg_max      = arg_int   (argc, argv, "--cg-max",      30000);
  const std::vector<double> hb_shifts =
      arg_double_csv(argc, argv, "--hasenbusch");
  const double hb_top      = arg_double(argc, argv, "--hasenbusch-top", 1.0);
  const double rat_lo      = arg_double(argc, argv, "--rat-lo",      1.0e-4);
  const double rat_hi      = arg_double(argc, argv, "--rat-hi",      200.0);
  const int    rat_degree  = arg_int   (argc, argv, "--rat-degree",  16);
  const int    rat_prec    = arg_int   (argc, argv, "--rat-precision",64);
  const int    n_traj      = arg_int   (argc, argv, "--n-trajectories", 20);
  const int    no_metrop   = arg_int   (argc, argv, "--no-metropolis-until", 0);
  const int    start_traj  = arg_int   (argc, argv, "--start-trajectory", 0);
  const std::string start_type =
      arg_str(argc, argv, "--starting-type", "ColdStart");
  const std::string ckpt_prefix =
      arg_str(argc, argv, "--ckpt-prefix", "ckpoint");
  const int    ckpt_interval = arg_int(argc, argv, "--ckpt-interval", 5);

  std::cout << GridLogMessage
            << "hmc_driver integrator=" << HMC_STRINGIFY(HMC_INTEGRATOR)
            << "  beta=" << beta
            << "  m_l="  << m_light  << "  m_s=" << m_strange
            << "  csw="  << csw
            << "  MDsteps=" << md_steps << "  trajL=" << traj_len
            << "  hb_rungs=" << hb_shifts.size()
            << std::endl;

  // --- HMC runner + resources --------------------------------------------
  using FermionImpl = WilsonImplR;
  using FermionAction = WilsonCloverFermionD;
  using FermionField = typename FermionAction::FermionField;

  typedef GenericHMCRunner<HMC_INTEGRATOR> HMCWrapper;
  HMCWrapper TheHMC;

  TheHMC.Resources.AddFourDimGrid("gauge");  // consumes --grid / --mpi

  CheckpointerParameters CPparams;
  CPparams.config_prefix = ckpt_prefix + "_lat";
  CPparams.rng_prefix    = ckpt_prefix + "_rng";
  CPparams.saveInterval  = ckpt_interval;
  CPparams.format        = "IEEE64BIG";
  TheHMC.Resources.LoadNerscCheckpointer(CPparams);

  RNGModuleParameters RNGpar;
  RNGpar.serial_seeds   = "1 2 3 4 5";
  RNGpar.parallel_seeds = "6 7 8 9 10";
  TheHMC.Resources.SetRNGSeeds(RNGpar);

  typedef PlaquetteMod<HMCWrapper::ImplPolicy> PlaqObs;
  TheHMC.Resources.AddObservable<PlaqObs>();

  auto GridPtr   = TheHMC.Resources.GetCartesian();
  auto GridRBPtr = TheHMC.Resources.GetRBCartesian();
  LatticeGaugeField U(GridPtr);  // placeholder; Grid populates from StartingType

  // --- Gauge action (Symanzik, will be stout-smeared at MD time) ---------
  typedef SymanzikGaugeAction<PeriodicGimplR> SymanzikR;
  SymanzikR GaugeAction(beta, u0);
  GaugeAction.is_smeared = true;

  // --- Light sector: Hasenbusch-preconditioned Nf=2 ----------------------
  // Build an ascending mass list [m_light, shifts..., hb_top], then stack
  // N ratios M(m_i)/M(m_{i+1}) and one plain action at the top mass.
  // Empty shifts → single plain action at m_light (no preconditioning).
  std::vector<double> light_masses;
  light_masses.push_back(m_light);
  for (double m : hb_shifts) light_masses.push_back(m);
  const bool use_ratio_stack = !hb_shifts.empty();
  if (use_ratio_stack) light_masses.push_back(hb_top);

  std::vector<std::unique_ptr<FermionAction>> light_ferm_ops;
  light_ferm_ops.reserve(light_masses.size());
  for (double m : light_masses) {
    light_ferm_ops.push_back(std::make_unique<FermionAction>(
        U, *GridPtr, *GridRBPtr, m, csw, csw));
  }

  ConjugateGradient<FermionField> LightCG(cg_tol, cg_max);

  // Non-EO forms: WilsonClover has gauge-dependent Mee (ConstEE()==0), which
  // the EO-preconditioned TwoFlavour actions assert against. Stock Grid's
  // Test_hmc_WilsonCloverFermionGauge.cc uses non-EO TwoFlavour for the same
  // reason. EO-preconditioned clover HMC requires the Schur-decomposed
  // actions from the TXQCD Grid fork, which are out of scope here.
  using LightRatioAction = TwoFlavourRatioPseudoFermionAction<FermionImpl>;
  std::vector<std::unique_ptr<LightRatioAction>> light_ratios;
  if (use_ratio_stack) {
    for (size_t i = 0; i + 1 < light_ferm_ops.size(); ++i) {
      light_ratios.push_back(std::make_unique<LightRatioAction>(
          *light_ferm_ops[i + 1], *light_ferm_ops[i], LightCG, LightCG));
      light_ratios.back()->is_smeared = true;
    }
  }

  // Top action: plain TwoFlavour at the heaviest mass (UV regulator).
  // When the ladder is empty this IS the light action at m_light.
  using LightTopAction = TwoFlavourPseudoFermionAction<FermionImpl>;
  auto &top_ferm = *light_ferm_ops.back();
  LightTopAction LightTop(top_ferm, LightCG, LightCG);
  LightTop.is_smeared = true;

  // --- Strange sector: Nf=1 via rational pseudofermion -------------------
  // Non-EO form: WilsonClover has gauge-dependent Mee (ConstEE()==0), which
  // OneFlavourEvenOddRational asserts against. The EO-preconditioned clover
  // RHMC lives in the TXQCD Grid fork and is out of scope here.
  FermionAction StrangeFerm(U, *GridPtr, *GridRBPtr, m_strange, csw, csw);
  OneFlavourRationalParams rat_params(rat_lo, rat_hi, cg_max, cg_tol,
                                      rat_degree, rat_prec,
                                      /*BoundsCheckFreq=*/100,
                                      /*mdtol=*/1.0e-6,
                                      /*BoundsCheckTol=*/1.0e-4);
  OneFlavourRationalPseudoFermionAction<FermionImpl>
      StrangeRat(StrangeFerm, rat_params);
  StrangeRat.is_smeared = true;

  // --- Action levels -----------------------------------------------------
  // Fermion level inner-stepped 1x; gauge level outer-stepped 4x.
  ActionLevel<HMCWrapper::Field> Level1(1);
  for (auto &r : light_ratios) Level1.push_back(r.get());
  Level1.push_back(&LightTop);
  Level1.push_back(&StrangeRat);

  ActionLevel<HMCWrapper::Field> Level2(4);
  Level2.push_back(&GaugeAction);

  TheHMC.TheAction.push_back(Level1);
  TheHMC.TheAction.push_back(Level2);

  // --- HMC parameters ----------------------------------------------------
  TheHMC.Parameters.MD.MDsteps        = md_steps;
  TheHMC.Parameters.MD.trajL          = traj_len;
  TheHMC.Parameters.Trajectories      = n_traj;
  TheHMC.Parameters.NoMetropolisUntil = no_metrop;
  TheHMC.Parameters.StartTrajectory   = start_traj;
  TheHMC.Parameters.StartingType      = start_type;
  TheHMC.Parameters.MetropolisTest    = true;
  TheHMC.Parameters.PerformRandomShift = false;

  // Allow --StartingType / --Trajectories / --StartingTrajectory /
  // --Thermalizations / --ParameterFile from Grid's own parser to override
  // the values we just set (matches Test_hmc_*.cc behaviour).
  TheHMC.ReadCommandLine(argc, argv);

  // --- Stout smearing policy --------------------------------------------
  Smear_Stout<HMCWrapper::ImplPolicy> Stout(stout_rho);
  SmearedConfiguration<HMCWrapper::ImplPolicy>
      SmearingPolicy(GridPtr, stout_nsmr, Stout);

  // --- Run ---------------------------------------------------------------
  TheHMC.Run(SmearingPolicy);

  std::cout << GridLogMessage << "hmc_driver complete." << std::endl;
  Grid_finalize();
  return 0;
}

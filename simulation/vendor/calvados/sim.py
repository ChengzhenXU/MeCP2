import numpy as np
import openmm
from openmm import unit

import mdtraj as md

import os

from calvados import build, interactions

from Bio.SeqUtils import seq3

from .components import *

class Sim:
    def __init__(self,path,config,components):
        """Construct CALVADOS systems from configuration and component dictionaries."""

        self.path = path
        # parse config
        for key, val in config.items():
            setattr(self, key, val)

        for key, val in components['defaults'].items():
            setattr(self, f'default_{key}', val)

        self.comp_dict = components['system']
        self.comp_defaults = components['defaults']

        self.box = np.array(self.box)
        self.eps_lj *= 4.184 # kcal to kJ/mol

        if self.restart == 'checkpoint' and os.path.isfile(f'{self.path}/{self.frestart}'):
            self.slab_eq = False
            self.bilayer_eq = False

        if self.slab_eq:
            self.rcent = interactions.init_slab_restraints(self.box,self.k_eq)

        if self.ext_force:
            self.rcent = openmm.CustomExternalForce(self.ext_force_expr)

    def make_components(self):
        self.components = np.empty(0)
        self.use_restraints = False
        # comp_setup = 'spiral' if self.topol=='shift_ref_bead' else 'linear'
        for name, properties in self.comp_dict.items():
            molecule_type = properties.get('molecule_type', self.default_molecule_type)
            if molecule_type == 'protein':
                # Protein component
                comp_setup = 'compact'
                comp = Protein(name, properties, self.comp_defaults)
            elif molecule_type in ['lipid','cooke_lipid']:
                # Lipid component
                comp_setup = 'linear'
                comp = Lipid(name, properties, self.comp_defaults)
            elif molecule_type in ['crowder']:
                # Crowder component
                comp_setup = 'compact'
                comp = Crowder(name, properties, self.comp_defaults)
            elif molecule_type in ['rna']:
                # Crowder component
                comp_setup = 'spiral'
                comp = RNA(name, properties, self.comp_defaults)
            elif molecule_type == 'cyclic':
                comp_setup = 'compact'
                comp = Cyclic(name, properties, self.comp_defaults)
            elif molecule_type == 'seastar':
                comp_setup = 'compact'
                comp = Seastar(name, properties, self.comp_defaults)
            elif molecule_type == 'ptm_protein':
                comp_setup = 'compact'
                comp = PTMProtein(name, properties, self.comp_defaults)
            else:
                # Generic component
                comp_setup = 'linear'
                comp = Component(name, properties, self.comp_defaults)

            comp.eps_lj = self.eps_lj
            comp.calc_properties(pH=self.pH, verbose=self.verbose, comp_setup=comp_setup)
            if comp.restraint:
                if comp.restraint_type == 'go':
                    comp.init_restraint_force(
                        eps_lj=self.eps_lj, cutoff_lj=self.cutoff_lj,
                        eps_yu=self.eps_yu, k_yu = self.k_yu
                    )
                else:
                    comp.init_restraint_force()
                self.use_restraints = True

            self.components = np.append(self.components, comp)

    def count_components(self):
        """ Count components and molecules. """

        self.ncomponents = 0
        self.nmolecules = 0

        for comp in self.components:
            self.ncomponents += 1
            self.nmolecules += comp.nmol

        print(f'Total number of components in the system: {self.ncomponents}')
        print(f'Total number of molecules in the system: {self.nmolecules}')

        # move lipids at the end of the array
        molecule_types = np.asarray([c.molecule_type for c in self.components])
        self.nlipids = np.sum([c.nmol if c.molecule_type == 'lipid' else 0 for c in self.components])
        self.ncookelipids = np.sum([c.nmol if c.molecule_type == 'cooke_lipid' else 0 for c in self.components])
        self.nproteins = np.sum([c.nmol if c.molecule_type == 'protein' else 0 for c in self.components])
        self.ncrowders = np.sum([c.nmol if c.molecule_type == 'crowder' else 0 for c in self.components])
        self.nrnas = np.sum([c.nmol if c.molecule_type == 'rna' else 0 for c in self.components])

        if ((self.ncomponents > 1) or (self.nmolecules > 1)) and (self.topol in ['single', 'center']):
            raise ValueError("Topol 'center' incompatible with multiple molecules.")

        # move proteins at the beginning of the array
        if self.nmolecules > self.nproteins:
            protein_components = self.components[np.where(molecule_types=='protein')]
            non_protein_components = self.components[np.where(molecule_types!='protein')]
            self.components = np.append(protein_components,non_protein_components)

    def build_system(self):
        """
        Set up system
        * component definitions
        * build particle coordinates
        * define interactions
        * set restraints
        """

        self.top = md.Topology()
        self.system = openmm.System()
        a, b, c = build.build_box(self.box[0],self.box[1],self.box[2])
        self.system.setDefaultPeriodicBoxVectors(a, b, c)


        # init interaction parameters (required before make components)
        self.eps_yu, self.k_yu = interactions.genParamsDH(self.temp,self.ionic)

        # make components
        self.make_components()
        self.count_components()

        # init interactions
        self.ah, self.yu = interactions.init_nonbonded_interactions(
            self.eps_lj,self.cutoff_lj,self.eps_yu,self.k_yu,self.cutoff_yu,self.fixed_lambda
            )
        if self.nlipids > 0:
            self.cos, self.cn = interactions.init_lipid_interactions(
            self.eps_lj,self.eps_yu,self.cutoff_yu,factor=1.9
            )
        if self.ncookelipids > 0:
            if self.nlipids > 0:
                raise
            self.cos, self.cn = interactions.init_lipid_interactions(
            self.eps_lj,self.eps_yu,self.cutoff_yu,factor=3.0
            )

        self.nparticles = 0 # bead counter
        self.grid_counter = 0 # molecule counter for xy and xyz grids

        self.pos = []

        if self.topol == 'slab': # proteins + rna
            self.xyzgrid = build.build_xyzgrid(self.nproteins+self.nrnas,[self.box[0],self.box[1],self.slab_width])
            self.xyzgrid += np.asarray([0,0,self.box[2]/2.-self.slab_width/2.])
            if self.ncrowders > 0: # crowder
                xyzgrid = build.build_xyzgrid(np.ceil(self.ncrowders/2.),[self.box[0],self.box[1],self.box[2]/2.-self.slab_outer])
                self.xyzgrid = np.append(self.xyzgrid, xyzgrid, axis=0)
                self.xyzgrid = np.append(self.xyzgrid, xyzgrid + np.asarray([0,0,self.box[2]/2.+self.slab_outer]), axis=0)
        elif self.topol == 'grid':
            self.xyzgrid = build.build_xyzgrid(self.nmolecules,self.box)
        if self.nlipids > 0:
            self.bilayergrid = build.build_xygrid(int(self.nlipids*1.05),self.box)
            if (self.nproteins + self.nrnas) > 0:
                xyzgrid = build.build_xyzgrid(np.ceil((self.nproteins+self.nrnas)/2.),[self.box[0],self.box[1],self.box[2]/2.-self.box[0]])
                self.xyzgrid = np.append(xyzgrid, xyzgrid + np.asarray([0,0,self.box[2]/2.+self.box[0]]), axis=0)
        if self.ncookelipids > 0:
            self.bilayergrid = build.build_xygrid(int(self.ncookelipids*1.05),self.box)
            if (self.nproteins + self.nrnas) > 0:
                xyzgrid = build.build_xyzgrid(np.ceil((self.nproteins+self.nrnas)/2.),[self.box[0],self.box[1],self.box[2]/2.-self.box[0]])
                self.xyzgrid = np.append(xyzgrid, xyzgrid + np.asarray([0,0,self.box[2]/2.+self.box[0]]), axis=0)

        for cidx, comp in enumerate(self.components):
            for idx in range(comp.nmol):
                if self.verbose:
                    print(f'Component {cidx}, Molecule {idx}: {comp.name}')
                # particle definitions
                self.add_mdtraj_topol(comp)
                self.add_particles_system(comp.mws)

                # add interactions + restraints
                if comp.molecule_type in ['protein','crowder','cyclic','seastar','ptm_protein']:
                    xs = self.place_molecule(comp)
                elif comp.molecule_type in ['lipid','cooke_lipid']:
                    xs = self.place_bilayer(comp)
                elif comp.molecule_type == 'rna':
                    xs = self.place_molecule(comp)
                self.add_interactions(comp)

                # add restraints towards box center
                if (self.slab_eq or self.ext_force) and comp.ext_restraint:
                    self.add_ext_restraints(comp)

        if self.custom_restraints:
            self.map_custom_restraints()
            self.add_custom_restraints()

        self.pdb_cg = f'{self.path}/top.pdb'
        a = md.Trajectory(self.pos, self.top, 0, self.box, [90,90,90])
        if self.restart != 'pdb': # only save new topology if no system pdb is given
            a.save_pdb(self.pdb_cg)

        self.add_forces_to_system()
        self.print_system_summary()

    def add_forces_to_system(self):
        """ Add forces to system. """

        # Intermolecular forces
        for force in [self.yu, self.ah]:
            self.system.addForce(force)

        if (self.nlipids > 0) or (self.ncookelipids > 0):
            for force in [self.cos, self.cn]:
                self.system.addForce(force)

        # Intramolecular forces
        for comp in self.components:
            comp.get_forces() # bonded, angles, restraints...
            for force in comp.forces:
                self.system.addForce(force)
            if comp.restraint:
                print(f'Number of restraints for comp {comp.name}: {comp.cs.getNumBonds()}')

        # External force
        if self.ext_force:
            self.system.addForce(self.rcent)

        # Equilibration forces
        if self.slab_eq:
            self.system.addForce(self.rcent)

        # Custom forces
        if self.custom_restraints:
            self.system.addForce(self.cres)
            print(f'Number of custom restraints: {self.cres.getNumBonds()}')

        # Barostat force
        if self.box_eq:
            barostat = openmm.openmm.MonteCarloAnisotropicBarostat(
                    [self.pressure[0]*unit.bar,self.pressure[1]*unit.bar,self.pressure[2]*unit.bar],
                    self.temp*unit.kelvin,self.boxscaling_xyz[0],self.boxscaling_xyz[1],
                    self.boxscaling_xyz[2],1000)
            self.system.addForce(barostat)

        # Bilayer eq. force
        if self.bilayer_eq:
            barostat = openmm.openmm.MonteCarloMembraneBarostat(self.pressure[0]*unit.bar,
                    0*unit.bar*unit.nanometer, self.temp*unit.kelvin,
                    openmm.openmm.MonteCarloMembraneBarostat.XYIsotropic,
                    openmm.openmm.MonteCarloMembraneBarostat.ZFixed, 10000)
            self.system.addForce(barostat)

    def print_system_summary(self, write_xml: bool = True):
        """ Print system information and write xml. """

        if write_xml:
            with open(f'{self.path}/{self.sysname}.xml', 'w') as output:
                output.write(openmm.XmlSerializer.serialize(self.system))

        print(f'{self.nparticles} particles in the system')
        print('---------- FORCES ----------')
        print(f'ah: {self.ah.getNumParticles()} particles, {self.ah.getNumExclusions()} exclusions')
        print(f'yu: {self.yu.getNumParticles()} particles, {self.yu.getNumExclusions()} exclusions')
        if self.slab_eq:
            print(f'Equilibration restraints (rcent) towards box center in z direction')
            print(f'rcent: {self.rcent.getNumParticles()} restraints')
        if self.bilayer_eq:
            print(f'Equilibration under zero lateral tension')
        if self.box_eq:
            print(f'Equilibration through changes in box side lengths along '+' and '.join(np.array(['X','Y','Z'])[self.boxscaling_xyz]))

    def place_molecule(self, comp: Component, ntries: int = 10000):
        """
        Place proteins based on topology.
        """

        if self.topol == 'slab':
            x0 = self.xyzgrid[self.grid_counter]
            # x0[2] = self.box[2] / 2. # center in z
            xs = x0 + comp.xinit
            self.grid_counter += 1
        elif self.topol == 'grid':
            x0 = self.xyzgrid[self.grid_counter]
            xs = x0 + comp.xinit
            self.grid_counter += 1
        elif self.topol == 'center':
            x0 = self.box * 0.5 # place in center of box
            xs = x0 + comp.xinit
        elif self.topol == 'shift_ref_bead':
            x0 = self.box * 0.5 # place in center of box
            xs = x0 + comp.xinit
            xs -= comp.xinit[self.ref_bead]
        else:
            xs = build.random_placement(self.box, self.pos, comp.xinit, ntries=ntries)
        for x in xs:
            self.pos.append(x)
            self.nparticles += 1
        return xs # positions of the comp (to be used for restraints)

    def place_bilayer(self, comp: Component, ntries: int = 10000):
        """
        Place proteins based on topology.
        """
        #print('bilayergrid.shape',self.bilayergrid.shape)
        inserted = False
        while not inserted:
            xs, inserted = build.build_xybilayer(self.bilayergrid[0], self.box, self.pos, comp.xinit)
            if not inserted:
                xs, inserted = build.build_xybilayer(self.bilayergrid[0], self.box, self.pos, comp.xinit, upward=False)
                idx = np.random.randint(self.bilayergrid.shape[0])
                self.bilayergrid[0] = self.bilayergrid[idx]
                self.bilayergrid = np.delete(self.bilayergrid,idx,axis=0)
        for x in xs:
            self.pos.append(x)
            self.nparticles += 1
        return xs # positions of the comp (to be used for restraints)

    def add_bonds(self, comp, offset):
        """ Add bond forces. """

        exclusion_map = comp.add_bonds(offset)
        self.add_exclusions(exclusion_map)

    def add_angles(self, comp, offset):
        """ Add bond forces. """

        exclusion_map = comp.add_angles(offset)
        self.add_exclusions(exclusion_map)

    def add_restraints(self, comp, offset, exclude_nonbonded = True):
        """ Add restraints to single molecule. """

        exclusion_map = comp.add_restraints(offset)
        if exclude_nonbonded: # exclude ah, yu when restraining
            self.add_exclusions(exclusion_map)

    def add_custom_restraints(self, exclude_nonbonded = True):
        exclusion_map = []
        # self.custom_restr_pairs = []
        self.cres = interactions.init_restraints(self.custom_restraint_type)
        for i, j, r, k in self.custom_restr_abs: # i, j, r, k
            self.cres, restr_pair = interactions.add_single_restraint(
                self.cres, self.custom_restraint_type, r, k, i, j)
            # self.custom_restr_pairs.append(restr_pair)
            exclusion_map.append([i,j])
        if exclude_nonbonded: # exclude cres when restraining
            self.add_exclusions(exclusion_map)

    def add_exclusions(self, exclusion_map):
        # exclude LJ, YU for restrained pairs
        for excl in exclusion_map:
            self.ah = interactions.add_exclusion(self.ah, excl[0], excl[1])
            self.yu = interactions.add_exclusion(self.yu, excl[0], excl[1])
            if self.nlipids > 0 or self.ncookelipids > 0:
                self.cos.addExclusion(excl[0], excl[1])
                self.cn.addExclusion(excl[0], excl[1])

    def add_interactions(self,comp):
        """
        Protein interactions for one molecule of composition comp
        """

        offset = self.nparticles - comp.nbeads # to get indices of current comp in context of system

        # Add Ashbaugh-Hatch
        for sig, lam in zip(comp.sigmas, comp.lambdas):
            if comp.molecule_type in ['lipid', 'cooke_lipid']:
                self.ah.addParticle([sig*unit.nanometer, lam, 0])
            elif comp.molecule_type == 'crowder':
                self.ah.addParticle([sig*unit.nanometer, lam, -1])
            else: # protein, RNA
                self.ah.addParticle([sig*unit.nanometer, lam, 1])
            if self.nlipids > 0 or self.ncookelipids > 0:
                if comp.molecule_type in ['lipid', 'cooke_lipid']:
                    self.cos.addParticle([sig*unit.nanometer, lam, 0])
                else:
                    self.cos.addParticle([sig*unit.nanometer, lam, 1])
        # Add Debye-Huckel
        for q in comp.qs:
            self.yu.addParticle([q])

        # Add Charge-Nonpolar Interaction
        if self.nlipids > 0 or self.ncookelipids > 0:
            id_cn = 1 if comp.molecule_type == 'protein' else -1
            for sig, alpha, q in zip(comp.sigmas, comp.alphas, comp.qs):
                self.cn.addParticle([(sig/2)**3, alpha, q, id_cn])

        # Add bonds
        self.add_bonds(comp, offset)

        if comp.molecule_type == 'rna':
            self.add_angles(comp, offset)

        # Add restraints
        if comp.restraint:
            self.add_restraints(comp,offset)

        # write lists
        if self.verbose:
            comp.write_bonds(self.path)
            if comp.restraint:
                comp.write_restraints(self.path)

    def add_ext_restraints(self,comp):
        """ Add external-potential restraints. """

        offset = self.nparticles - comp.nbeads # to get indices of current comp in context of system
        for i in range(0,comp.nbeads):
            self.rcent.addParticle(i+offset)

    def add_mdtraj_topol(self, comp):
        """ Add one molecule to mdtraj topology. """

        # Note: Move this to component eventually.
        chain = self.top.add_chain()

        if comp.molecule_type == 'rna':
            for idx,resname in enumerate(comp.seq):
                res = self.top.add_residue(resname, chain, resSeq=idx+1)
                self.top.add_atom(resname+"P", element=md.element.phosphorus, residue=res)
                self.top.add_atom(resname+"N", element=md.element.nitrogen, residue=res)
            for i in range(comp.nbeads-1):
                for j in range(1,comp.nbeads):
                    if comp.bond_check(i,j):
                        self.top.add_bond(chain.atom(i), chain.atom(j))
        else:
            for idx,resname in enumerate(comp.seq):
                if comp.molecule_type in ['protein','crowder']:
                    resname = comp.residues.loc[resname,'three']
                res = self.top.add_residue(resname, chain, resSeq=idx+1)
                self.top.add_atom('CA', element=md.element.carbon, residue=res)
            for i in range(chain.n_atoms-1):
                if comp.bond_check(i,i+1):
                    self.top.add_bond(chain.atom(i), chain.atom(i+1))

    def add_particles_system(self,mws):
        """ Add particles of one molecule to openMM system. """

        for mw in mws:
            self.system.addParticle(mw*unit.amu)

    def map_custom_restraints(self):
        """ Map input format for custom restraints to absolute bead number """
        custom_restr = self.parse_custom_restraints(self.fcustom_restraints)
        total_beads = [0]
        for idx, comp in enumerate(self.components):
            comp.start_bead = total_beads[-1]
            total_beads.append(int(comp.nmol * comp.nbeads))
        self.custom_restr_abs = []
        for i,j,r,k in custom_restr:
            print(i,j,r,k)
            crestr = []
            for idx, x in enumerate([i,j]):
                name, copy, bead = x[0], x[1], x[2] # 1-based
                for idx, comp in enumerate(self.components):
                    if comp.name == name:
                        x_abs = comp.start_bead + (copy-1)*comp.nbeads + (bead-1)
                        break
                crestr.append(x_abs)
            crestr.append(float(r))
            crestr.append(float(k))
            self.custom_restr_abs.append(crestr)

    @staticmethod
    def parse_custom_restraints(fcustom_restraints):
        custom_restraints = []
        with open(fcustom_restraints,'r') as f:
            for line in f.readlines():
                spl = line.split('|')
                i = spl[0].split()
                j = spl[1].split()
                r = spl[2].split()[0]
                k = spl[2].split()[1]
                restr = [
                    [i[0], int(i[1]), int(i[2])],
                    [j[0], int(j[1]), int(j[2])],
                    r,
                    k
                ]
                custom_restraints.append(restr) # 1-based
        return custom_restraints

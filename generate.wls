#!/usr/bin/env wolframscript
Import /@ {"model/cell.mx", "model/mat.mx"};

BuildCell[Lx_, Ly_, Lz_] := (
    centeredcell = Select[centeredcell, #[[1]]!=1&];
    supercell = {#[[1]] // Round, #[[2 ;; 4]]} & /@ Partition[Table[(({
         {1, 0, 0, 0, 0},
         {0, 1, 0, 0, x (9.99816)},
         {0, 0, 1, 0, y (9.99816)},
         {0, 0, 0, 1, z (9.99816)},
         {0, 0, 0, 0, 1}
        })) . i, {i, centeredcell}, {x, -Lx, Lx}, {y, -Ly, Ly}, {z, -Lz, Lz}] // Flatten, 5];
      
    newsupercell = supercell;    
    lostspins = {Length[supercell]-Length[newsupercell], Length[supercell]};
    supercell = newsupercell;

    If[sphere != 0,
        center = Mean@(supercell[[All,2]]);
        Print["center: "<>ToString[center]];
        supercell = {#[[1]], #[[2]]-center}&/@supercell;

        radius = Max[MaximalBy[supercell[[All,2]], First]//First//First, MaximalBy[supercell[[All,2]], Last]//First//Last];
        Print["radius: "<>ToString[radius]];

        supercell = If[Norm[#[[2]]]<radius, {#[[1]],#[[2]]}, Missing[]]&/@supercell;
        supercell = supercell//DeleteMissing;
    ];

    Clear[newsupercell];
    StringTemplate["We lost `` out of ``"]@@lostspins//logWrite
);

PrepareSpins := (
    poso = Join[#[[2]], {If[#[[1]] == 2, -1.0, 1.0]}] & /@ supercell;
    spino = Join[Normalize@RandomReal[{-1, 1}, 3], {0}] & /@ supercell;
    
);

eta = 0.06483931947069943`; lambda = -70.`; rhos = 0.18`; v = 50.0; delta = 2300.`; mu = 0.47`; g = 2.029;

sphere = 0;

hitemp = 200;
lotemp = 10;
cycles = 200;

maxcycles = 1000;

width = 300;
height = 300;
df = 7.0;
att = 20.0;

comm = StringTake[CreateUUID[],4]; 

dip = 1.75;
jxx = 0.0;
jyy = -10.3;
jxy = 14.39;
tresh = 18.49;

argv = Rest @ $ScriptCommandLine;
argc = Length @ argv;
Do[Print[argv[[i]]], {i, argc}];

Do[ToExpression[argv[[i]]], {i, 2, argc}];

size = ToExpression/@StringSplit[argv[[1]], "x"];
BuildCell@@size;
PrepareSpins;



dirname = "projects/"<>argv[[1]]<>"-"<>ToString[comm];
CreateDirectory[dirname];
CreateDirectory[dirname<>"/snapshots"];

ConstA = v U\[Theta] + (-(rhos + lambda^2/delta + 2 eta) ((3 Sz . Sz - ee 2 (2 + 1)) . U\[Theta] + Sqrt[3] (Sx . Sx - Sy . Sy) . U\[CurlyEpsilon])) // N;

Export[dirname<>"/static_a.bin", {Re[#], Im[#]}&/@(ConstA//Flatten)//Flatten//Evaluate, "Real32"];
Export[dirname<>"/sx_a.bin", {Re[#], Im[#]}&/@(Sx//Flatten)//Flatten//Evaluate, "Real32"];
Export[dirname<>"/sy_a.bin", {Re[#], Im[#]}&/@(Sy//Flatten)//Flatten//Evaluate, "Real32"];
Export[dirname<>"/sz_a.bin", {Re[#], Im[#]}&/@(Sz//Flatten)//Flatten//Evaluate, "Real32"];

Export[dirname<>"/init.bin", spino//Flatten, "Real32"];
Export[dirname<>"/coordinates.bin", poso//Flatten, "Real32"];

Export[dirname<>"/cmd.txt", StringJoin[Riffle[$ScriptCommandLine, " "]], "String"];

NS = spino//Length;
PT = Ceiling[NS/512//N];

Run[StringTemplate["nohup ./sim `` `` `` `` `` `` `` `` `` `` `` `` `` `` `` `` > ``/logs.txt &"][dirname, NS, PT, dip, jxx, jyy, jxy, tresh, hitemp, lotemp, cycles, width, height, df, att, maxcycles, dirname]];
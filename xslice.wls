#!/usr/bin/env wolframscript

x2da[z_:2] := (
    Graphics[{Arrowheads[
         0.02], ({RGBColor[
             Round[Abs@Normalize@Re@Take[spino[[#[[1]]]], 3], 0.0001]], 
            Arrow[{#[[2]][[
               1 ;; 2]], (#[[2]] + 1.2 Re@spino[[#[[1]]]])[[
               1 ;; 2]]}]} & /@ 
          Select[Transpose[{Table[i, {i, 1, Length[poso]}], poso}], 
           Abs[#[[2]][[3]] - z] < 2 &])}]
);

x2d[z_:2] := (
    Graphics[{Arrowheads[
         0.02], ({ Hue[ToPolarCoordinates[(Re@Take[spino[[#[[1]]]], 2])][[2]] / (2 Pi)], 
            Arrow[{#[[2]][[
               1 ;; 2]], (#[[2]] + 1.2 Re@spino[[#[[1]]]])[[
               1 ;; 2]]}]} & /@ 
          Select[Transpose[{Table[i, {i, 1, Length[poso]}], poso}], 
           (Abs[#[[2]][[3]] - z] < 2 && #[[2,4]]>0.0) &])}]
);

dwg[z_:2] := (
    Graphics[{Arrowheads[
         0.02], ({RGBColor[{(1+poso[[ #[[1]], 4 ]])/2,(1-poso[[ #[[1]], 4 ]])/2,(1-poso[[ #[[1]], 4 ]])/2}], 
            Arrow[{#[[2]][[
               1 ;; 2]], (#[[2]] + 1.2 Re@spino[[#[[1]]]])[[
               1 ;; 2]]}]} & /@ 
          Select[Transpose[{Table[i, {i, 1, Length[poso]}], poso}], 
           Abs[#[[2]][[3]] - z] < 2 &])}]
);

width = 300;
height = 300;

z = 2;

ext = "png";
comm = StringTake[CreateUUID[],3];


fun = x2d;

cycle = Last[];

argv = Rest @ $ScriptCommandLine;
argc = Length @ argv;
Do[Print[argv[[i]]], {i, argc}];

Do[ToExpression[argv[[i]]], {i, 2, argc}];

dirname = "projects/"<>argv[[1]];

If[!DirectoryQ[dirname],
    dirname = "tests/"<>argv[[1]];
];

If[!DirectoryQ[dirname],
    Print["doesnt exist"];
    Exit[-1];
];

CreateDirectory[dirname<>"/xslice"];

If[cycle === Last[],
    cycle = (ToExpression/@(StringDrop[#, 6]&/@(FileBaseName/@FileNames["*.bin", dirname<>"/snapshots"])))//Max;
];

If[cycle === all,
    Do[
        spino = Partition[Import[dirname<>"/snapshots/cycle_"<>ToString[cycle]<>".bin",  "Real32"],4];
        poso = Partition[Import[dirname<>"/coordinates.bin", "Real32"], 4];
        Export[dirname<>"/xslice/"<>ToString[fun]<>"-"<>ToString[cycle]<>"-z"<>ToString[z//Round]<>"-"<>ToString[comm]<>"."<>ext, Show[fun[z], ImageSize->width]];
        
    , {cycle, (ToExpression/@(StringDrop[#, 6]&/@(FileBaseName/@FileNames["*.bin", dirname<>"/snapshots"])))//Sort}]
,
    spino = Partition[Import[dirname<>"/snapshots/cycle_"<>ToString[cycle]<>".bin",  "Real32"],4];
    poso = Partition[Import[dirname<>"/coordinates.bin", "Real32"], 4];

    Export[dirname<>"/xslice/"<>ToString[fun]<>"-"<>ToString[cycle]<>"-z"<>ToString[z//Round]<>"-"<>ToString[comm]<>"."<>ext, Show[fun[z], ImageSize->width]];
];

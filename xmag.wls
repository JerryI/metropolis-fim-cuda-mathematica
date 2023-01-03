#!/usr/bin/env wolframscript

cycle = last[];

argv = Rest @ $ScriptCommandLine;
argc = Length @ argv;
Do[Print[argv[[i]]], {i, argc}];

Do[ToExpression[argv[[i]]], {i, 2, argc}];

dirname = "projects/"<>argv[[1]];

If[cycle === last[],
    cycle = (ToExpression/@(StringDrop[#, 6]&/@(FileBaseName/@FileNames["*.bin", dirname<>"/snapshots"])))//Max;
];

spino = Partition[Import[dirname<>"/snapshots/cycle_"<>ToString[cycle]<>".bin",  "Real32"],4];
poso = Partition[Import[dirname<>"/coordinates.bin", "Real32"], 4];

mag = 2.0*3.0*Total[spino]/Length[spino];

Print[StringTemplate["average manetization is ``; ``; ``"]@@mag];


{
  description = "Virtual Environment for Python";
  inputs.systems.url = "github:nix-systems/default";

  outputs = {
    self,
    nixpkgs,
    flake-utils,
    systems,
  }:
    flake-utils.lib.eachSystem (import systems)
    (system: let
      pkgs = import nixpkgs {
        inherit system;
      };
      lib = pkgs.lib;
    in {
      packages = flake-utils.lib.flattenTree {
        inherit (pkgs) hello;
      };

      devShells.default = let
        pythonPackages = pkgs.python39Packages;
      in pkgs.mkShell rec {
        venvDir = "./.venv";
        NIX_LD_LIBRARY_PATH = lib.makeLibraryPath (with pkgs; [
          stdenv.cc.cc.lib
          stdenv.cc.cc
          stdenv.cc
          qt6.qtbase
          qt6.qtsvg
          qt6.qtdeclarative
          qt6.qtwayland
          libcxx
          openssl
          zlib
        ]);
        NIX_LD = pkgs.runCommand "ld.so" {} ''
          ln -s "$(cat '${pkgs.stdenv.cc}/nix-support/dynamic-linker')" $out
        '';
        LD_LIBRARY_PATH = NIX_LD_LIBRARY_PATH;
        QT_PLUGIN_PATH = "${pkgs.qt6.qtbase}/${pkgs.qt6.qtbase.qtPluginPrefix}:${pkgs.qt6.qtwayland}/${pkgs.qt6.qtbase.qtPluginPrefix}";
        # lib.fileContents "${pkgs.stdenv.cc}/nix-support/dynamic-linker";
        buildInputs = (with pkgs; [
          pythonPackages.python
          pythonPackages.venvShellHook
          stdenv.cc
          stdenv.cc.cc.lib
          openssl
          swig
          zlib
        ]) ++ (with pythonPackages; [
          python
          venvShellHook
          tkinter
        ]);
      };
    });
}

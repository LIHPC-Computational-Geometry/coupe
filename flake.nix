{
  inputs = {
    fenix = {
      url = "github:nix-community/fenix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    flake-utils.url = "github:numtide/flake-utils";
    nixpkgs.url = "nixpkgs/nixos-unstable";
  };

  outputs = { nixpkgs, flake-utils, fenix, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        overlays = [ fenix.overlays.default ];
        pkgs = import nixpkgs { inherit system overlays; };

        
        commonBuildInputs = with pkgs; [
          # Rust
          (fenix.packages.${system}.stable.withComponents [
            "cargo"
            "clippy"
            "rust-src"
            "rustc"
            "rustfmt"
          ])

          # Tools
          cargo-nextest
          taplo
        ];
        linuxBuildInputs = with pkgs; [
        ];
        darwinBuildInputs = with pkgs; [
          libiconv
        ];

        buildInputs = commonBuildInputs
          ++ (if pkgs.stdenv.isLinux  then linuxBuildInputs  else [])
          ++ (if pkgs.stdenv.isDarwin then darwinBuildInputs else []);

        ldLibraryPath = pkgs.lib.makeLibraryPath ( buildInputs );
      in {
        devShell = pkgs.mkShell {
          buildInputs = buildInputs;
        };
      });
}

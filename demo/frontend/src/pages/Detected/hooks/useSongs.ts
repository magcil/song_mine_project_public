import { useLayoutEffect } from "react";
import { useStores } from "stores/rootStore";

export const useSongs = () => {
  const { songsStore, functionalityStore } = useStores();
  useLayoutEffect(() => {
    functionalityStore.triggerActivity(true);
    songsStore.getAllSongs().finally(() => {
      functionalityStore.triggerActivity(false);
    });
  }, []);
};

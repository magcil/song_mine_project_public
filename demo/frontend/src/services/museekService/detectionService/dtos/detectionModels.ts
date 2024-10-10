export interface ResultDTO {
  winner: string;
  score: number;
  datetime: string;
  device: string;
}

export interface PaginatedResultDTO {
  totalCount: number;
  data: ResultDTO[];
  page: number;
  page_size: number;
}

export interface PopularWinnerDTO {
  winner: string;
  occurrences: number;
}

export interface TopWinnersByDeviceDTO {
  device: string;
  winners: { [key: string]: number };
}

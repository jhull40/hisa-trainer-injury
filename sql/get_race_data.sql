SELECT
      main.[ChartID]
      ,[Charts_TracksRaceDatesID]
      ,RaceDate as race_date
      ,TrackID as track_id
      ,[RaceNumber] as race_number
      ,[RacePurse] as purse
      ,[ClaimingRace] as claiming_race
      ,[RaceCourse] as surface
      ,[DistanceAbbreviated] as distance
      ,[TrackCondition] as track_condition
      ,[ClassRating] as class_rating
      ,[RaceTypeLong] as race_type
      ,[ScratchedHorsesDetail] as scratches
      ,[Charts_HorsesID]
      ,main.[ChartID]
      ,horses.[RegistrationNumber] as registration_number
      ,[HorseName] as horse_name
      ,bio.sex
      ,[TrainerName] as trainer_name
      ,[ProgramNumber] as program_number
      ,[PostPosition] as post_position
      ,[PositionAtFinish] as finish
      ,[Lasix] as lasix
      ,[LasixRed] as lasix_red
      ,[Wraps] as wraps
      ,[Blinkers] as blinkers
      ,[BlinkersRed] as blinkers_red
      ,[ClosingOdds] as odds
      ,[PerformanceFigure] as performance_figure
      ,case when PerformanceFigure = '-' then 1 else 0 end as dnf
      ,[LongComment] as comment
      ,[FinishPosition_Behind] as finish_position_behind
      ,foaling_date
      ,DATEDIFF(day,foaling_date, RaceDate)/365.0 as age
      ,case when date_of_death = '1900-01-01 00:00:00.000' then null else date_of_death end as date_of_death
  FROM 
  [TFUS_Prod].[dbo].[Charts_Horses] horses 
  INNER JOIN [RTR_Prod].[dbo].[horse] bio
  on horses.RegistrationNumber = bio.registration_number

  inner join [TFUS_Prod].[dbo].[Charts_Main] main
  on main.ChartID = horses.ChartID

  inner join [TFUS_Prod].[dbo].[TracksRaceDates] tracks
  on main.Charts_TracksRaceDatesID = tracks.TrackRaceDateID
  
where left(TrainerName, 1) = 'A'
--registration_number = '00036005'
--order by registration_number, race_date
--where left(TrainerName, 1) in ('H', 'I', 'J', 'K', 'L')
--('M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z')
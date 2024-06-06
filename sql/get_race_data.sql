SELECT 
r.race_date,
r.track_id,
r.race_number,
--r.country,
race_type,
--age_restriction,
--sex_restriction,
--stakes_indicator,
r.distance_id,
r.distance_unit,
r.surface,
course_type,
--purse_usa as purse,
track_condition,
weather,
--chute_start,
track_sealed_indicator,
s.registration_number,
medication,
equipment,
weight_carried,
horse_weight,
--program_number,
--post_position,
--position_at_start,
--position_at_point_of_call_1,
--position_at_point_of_call_2,
--position_at_point_of_call_3,
--position_at_point_of_call_4,
--position_at_point_of_call_5,
--position_at_finish,
official_position,
--dead_heat_indicator,
--length_ahead_at_poc_1,
--length_ahead_at_poc_2,
--length_ahead_at_poc_3,
--length_ahead_at_poc_4,
--length_ahead_at_poc_5,
--length_ahead_at_finish,
--length_behind_at_poc_1,
--length_behind_at_poc_2,
--length_behind_at_poc_3,
--length_behind_at_poc_4,
--length_behind_at_poc_5,
length_behind_at_finish,
jockey_id,
trainer_id,
owner_id,
trouble_indicator,
scratch_indicator,
scratch_reason,
short_comment,
long_comment,
horse_name,
--foaling_date,
case 
  when sex_change_reported_date is null then sex
  when sex_change_reported_date > r.race_date then sex_prior_to_change
  else sex 
  end as sex,

--sex_change_reported_date,
--sex_prior_to_change,
DATEDIFF(day,foaling_date, r.race_date)/365.0 as age
--case when date_of_death = '1900-01-01 00:00:00.000' then null else date_of_death end as date_of_death 


  FROM  [RTR_Prod].[dbo].[race] r
  inner join [RTR_Prod].[dbo].[start] s
  on s.track_id = r.track_id 
  and s.race_date = r.race_date
  and s.race_number = r.race_number
  inner join [RTR_Prod].[dbo].[horse] h
  on h.registration_number = s.registration_number
--inner join [RTR_Prod].[dbo].[distance] d
--on d.distance_id = r.distance_id and r.breed_type = d.breed_type
  
where 
--left(horse_name,1) = 'B'
YEAR(r.race_date) = 2023